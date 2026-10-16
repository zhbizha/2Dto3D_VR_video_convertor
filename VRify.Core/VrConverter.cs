using System.Threading.Channels;

namespace VRify.Core;

public enum StereoMode
{
    /// <summary>Flat side-by-side pair (TV-style 3D).</summary>
    FlatSbs,
    /// <summary>VR180 equirectangular SBS — for headset players in VR180 mode.</summary>
    Vr180,
}

public sealed record ConversionSettings(
    string InputPath,
    string OutputPath,
    float DepthStrengthPercent = 2.5f,
    float TemporalSmoothing = 0.6f,
    int Crf = 16,
    bool Hevc = true,
    int? MaxBitrateKbps = null,
    int GpuDeviceId = 0,
    StereoMode Mode = StereoMode.FlatSbs,
    double ScreenFovDeg = 160.0,
    int EyeSize = 0);          // 0 = auto (2× source height, even, ≤2160)

public sealed record ConversionProgress(long FramesDone, long FramesTotal, double Fps)
{
    public double Percent => FramesTotal <= 0 ? 0 : Math.Min(100, 100.0 * FramesDone / FramesTotal);
}

/// <summary>Result of a single-frame preview conversion.</summary>
public sealed record PreviewResult(byte[] Sbs, int Width, int Height);

public sealed class VrConverter : IDisposable
{
    public const string DefaultModelFileName = "model.onnx";

    private readonly DepthEstimator _depth;
    private readonly GpuPipeline? _gpu;
    private readonly string _gpuInfo;

    public string Provider => _depth.Provider;

    /// <summary>Exposed for diagnostics; production code must not poke at the depth estimator directly.</summary>
    public DepthEstimator Depth => _depth;

    /// <summary>GPU compute pipeline info, or the reason the CPU fallback is in use.</summary>
    public string GpuInfo => _gpuInfo;

    public VrConverter(string modelPath, int gpuDeviceId = 0)
    {
        _depth = new DepthEstimator(modelPath, gpuDeviceId);
        _gpu = GpuPipeline.TryCreate(gpuDeviceId, out _gpuInfo);

        // A/B switches used by the probe harness to reproduce the pre-fix behaviour.
        if (Environment.GetEnvironmentVariable("VRIFY_NO_SHARPEN") is { Length: > 0 })
            DepthEstimator.SharpenEdges = false;
    }

    /// <summary>Legacy fill = thin-row interpolation + push-pull blur (pre-fix behaviour).</summary>
    private static bool UseBackgroundFill =>
        Environment.GetEnvironmentVariable("VRIFY_LEGACY_FILL") is not { Length: > 0 };

    /// <summary>Trend fill = destination-domain mirror that continues the background's slope.</summary>
    private static bool UseTrendFill =>
        Environment.GetEnvironmentVariable("VRIFY_TREND_FILL") is { Length: > 0 };

    /// <summary>Resolves the bundled model path, preferring fp16 (faster on GPU).</summary>
    public static string FindModel()
    {
        var names = new[] { "model_fp16.onnx", DefaultModelFileName };
        var roots = new[]
        {
            AppContext.BaseDirectory,
            Path.Combine(AppContext.BaseDirectory, "models"),
            Environment.CurrentDirectory,
            Path.Combine(Environment.CurrentDirectory, "models"),
        };
        foreach (var n in names)
            foreach (var r in roots)
            {
                var p = Path.Combine(r, n);
                if (File.Exists(p)) return p;
            }
        throw new FileNotFoundException(
            "Depth model not found. Expected model.onnx (or model_fp16.onnx) next to the executable.",
            Path.Combine(AppContext.BaseDirectory, DefaultModelFileName));
    }

    /// <summary>Runs the full 2D → SBS conversion (flat or VR180) with progress reporting.</summary>
    public void Convert(ConversionSettings s, IProgress<ConversionProgress>? progress = null,
        CancellationToken ct = default)
    {
        var info = VideoInfo.Probe(s.InputPath);
        int w = info.Width, h = info.Height;
        int frameSize = w * h * 3;

        bool vr = s.Mode == StereoMode.Vr180;
        int eyeSize = s.EyeSize > 0 ? s.EyeSize : Math.Clamp(EvenUp(h * 2), 640, 2160);
        int outW, outH;
        if (vr) { outW = eyeSize * 2; outH = eyeSize; }
        else    { outW = w * 2;        outH = h; }

        bool useGpu = _gpu != null;
        if (useGpu)
        {
            _gpu!.BackgroundFill = UseBackgroundFill;
            _gpu!.Configure(w, h, vr, eyeSize, s.ScreenFovDeg,
                s.DepthStrengthPercent, s.TemporalSmoothing);
        }
        int sbsSize = outW * outH * (useGpu ? 4 : 3);   // GPU path outputs RGBA

        var reader = new FrameReader(s.InputPath, w, h);
        var writer = new FrameWriter(s.OutputPath, info, outW, outH, info.FrameRate,
            s.Crf, s.Hevc, s.MaxBitrateKbps, inputRgba: useGpu);
        var synth = new StereoSynthesizer(s.DepthStrengthPercent, temporalSmoothing: 0f)
        {
            BackgroundFill = UseBackgroundFill,
            TrendFill = UseTrendFill,
        };
        var smoother = new DepthSmoother(s.TemporalSmoothing);
        var projector = vr ? new EquirectProjector(s.ScreenFovDeg) : null;

        // ---- 4-stage pipeline: decode → depth(+smooth+refine) → warp → encode.
        // Stages run concurrently (GPU inference overlaps CPU warp/decode/encode);
        // bounded queues give backpressure; pools of big buffers → zero per-frame GC.
        var rgbPool = CreatePool(6, () => new byte[frameSize]);
        var depthPool = CreatePool(6, () => new float[w * h]);
        var sbsPool = CreatePool(4, () => new byte[sbsSize]);
        var qDecode = Channel.CreateBounded<byte[]>(new BoundedChannelOptions(4));
        var qDepth = Channel.CreateBounded<(byte[] rgb, float[] depth)>(
            new BoundedChannelOptions(4));
        var qWarp = Channel.CreateBounded<byte[]>(new BoundedChannelOptions(4));

        long done = 0;
        int finished = 0;
        Exception? stageEx = null;
        var wall = System.Diagnostics.Stopwatch.StartNew();
        long tDecode = 0, tDepth = 0, tSmooth = 0, tRefine = 0, tWarp = 0, tEncode = 0;

        void CompleteAll(Exception ex)
        {
            qDecode.Writer.TryComplete(ex);
            qDepth.Writer.TryComplete(ex);
            qWarp.Writer.TryComplete(ex);
            rgbPool.Writer.TryComplete(ex);
            depthPool.Writer.TryComplete(ex);
            sbsPool.Writer.TryComplete(ex);
        }

        void Fail(Exception ex)
        {
            // Unwrap channel wrappers so the first REAL error wins the race.
            if (ex is ChannelClosedException { InnerException: { } inner }) ex = inner;
            if (Interlocked.CompareExchange(ref stageEx, ex, null) == null)
                CompleteAll(ex);
        }

        using var reg = ct.Register(() =>
        {
            if (Volatile.Read(ref finished) == 0)
                Fail(new OperationCanceledException(ct));
        });

        async Task DecodeStage()
        {
            try
            {
                while (true)
                {
                    byte[] rgb = await rgbPool.Reader.ReadAsync();
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    int n = reader.ReadFrame(rgb);
                    Interlocked.Add(ref tDecode, sw.ElapsedTicks);
                    if (n != frameSize)
                    {
                        rgbPool.Writer.TryWrite(rgb);   // unused
                        break;                          // EOF
                    }
                    await qDecode.Writer.WriteAsync(rgb);
                }
            }
            catch (Exception ex) { Fail(ex); }
            finally { qDecode.Writer.TryComplete(); }
        }

        async Task DepthStage()
        {
            try
            {
                await foreach (var rgb in qDecode.Reader.ReadAllAsync())
                {
                    float[] dst = await depthPool.Reader.ReadAsync();

                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    float[] raw = _depth.EstimateDepth(rgb, w, h, w * 3);
                    Interlocked.Add(ref tDepth, sw.ElapsedTicks);

                    if (useGpu)
                    {
                        // GPU pipeline does smooth + refine + warp itself (serially in WarpStage)
                        Array.Copy(raw, dst, raw.Length);
                        await qDepth.Writer.WriteAsync((rgb, dst));
                        continue;
                    }

                    sw.Restart();
                    float[] sm = smoother.Smooth(raw, rgb, w);
                    Interlocked.Add(ref tSmooth, sw.ElapsedTicks);

                    // Re-snap: temporal blending smears the depth transition band where
                    // the silhouette moved between frames; re-running the guided filter
                    // with the CURRENT frame as guide pulls edges back onto their true
                    // position. This is the main anti-"swimming" step.
                    sw.Restart();
                    _depth.RefineEdges(sm, rgb, w, h, w * 3);
                    Interlocked.Add(ref tRefine, sw.ElapsedTicks);

                    // Copy into pooled buffer: estimator/smoother reuse their own storage
                    // for the next frame, but warp stage reads this one concurrently.
                    Array.Copy(sm, dst, sm.Length);
                    await qDepth.Writer.WriteAsync((rgb, dst));
                }
            }
            catch (Exception ex) { Fail(ex); }
            finally { qDepth.Writer.TryComplete(); }
        }

        async Task WarpStage()
        {
            byte[]? flat = vr ? new byte[frameSize * 2] : null;
            try
            {
                await foreach (var (rgb, depth) in qDepth.Reader.ReadAllAsync())
                {
                    byte[] sbs = await sbsPool.Reader.ReadAsync();

                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    if (useGpu)
                    {
                        // Full post-depth chain on GPU: smooth → refine → warp →
                        // thin-fill → pyramid inpaint → pack/project → readback.
                        _gpu!.Render(rgb, depth, sbs);
                    }
                    else if (!vr)
                    {
                        synth.Synthesize(rgb, depth, w, h, sbs);
                    }
                    else
                    {
                        synth.Synthesize(rgb, depth, w, h, flat!);
                        projector!.Project(flat!, 0, w, h, w * 2 * 3,
                            sbs, 0, eyeSize * 2, eyeSize, eyeSize);
                        projector.Project(flat!, w, w, h, w * 2 * 3,
                            sbs, eyeSize, eyeSize * 2, eyeSize, eyeSize);
                    }
                    Interlocked.Add(ref tWarp, sw.ElapsedTicks);

                    // Source frames no longer needed → recycle BEFORE handing off,
                    // so the decoder can run ahead while encoder drains.
                    rgbPool.Writer.TryWrite(rgb);
                    depthPool.Writer.TryWrite(depth);
                    await qWarp.Writer.WriteAsync(sbs);
                }
            }
            catch (Exception ex) { Fail(ex); }
            finally { qWarp.Writer.TryComplete(); }
        }

        async Task EncodeStage()
        {
            double fpsEma = 0;
            try
            {
                await foreach (var sbs in qWarp.Reader.ReadAllAsync())
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    writer.WriteFrame(sbs);
                    Interlocked.Add(ref tEncode, sw.ElapsedTicks);
                    sbsPool.Writer.TryWrite(sbs);

                    long d = Interlocked.Increment(ref done);
                    double inst = d / Math.Max(0.001, wall.Elapsed.TotalSeconds);
                    fpsEma = fpsEma == 0 ? inst : fpsEma * 0.9 + inst * 0.1;
                    progress?.Report(new ConversionProgress(d, info.FrameCount, fpsEma));
                }
            }
            catch (Exception ex) { Fail(ex); }
        }

        var stages = new[] { DecodeStage(), DepthStage(), WarpStage(), EncodeStage() };
        try { Task.WhenAll(stages).GetAwaiter().GetResult(); }
        catch { /* all stage exceptions already captured via Fail */ }
        Volatile.Write(ref finished, 1);

        // Flush/close ffmpeg processes; prefer the first pipeline error over flush noise.
        try { writer.Dispose(); } catch (Exception ex) { if (stageEx == null) stageEx = ex; }
        try { reader.Dispose(); } catch (Exception ex) { if (stageEx == null) stageEx = ex; }

        if (ct.IsCancellationRequested) throw new OperationCanceledException(ct);
        if (stageEx != null)
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(stageEx).Throw();

        long frames = Interlocked.Read(ref done);
        if (frames > 0)
        {
            double wallMs = wall.Elapsed.TotalMilliseconds / frames;
            LastStageProfile = string.Join("  ", new[]
            {
                $"wall={wallMs:0.0}",
                useGpu
                    ? $"(decode={Ms(tDecode, frames):0.0} depth={Ms(tDepth, frames):0.0} " +
                      $"gpu={Ms(tWarp, frames):0.0} encode={Ms(tEncode, frames):0.0})"
                    : $"(decode={Ms(tDecode, frames):0.0} depth={Ms(tDepth, frames):0.0} " +
                      $"smooth={Ms(tSmooth, frames):0.0} refine={Ms(tRefine, frames):0.0} " +
                      $"warp={Ms(tWarp, frames):0.0} encode={Ms(tEncode, frames):0.0})",
            });
        }
    }

    private static Channel<T> CreatePool<T>(int count, Func<T> factory)
    {
        var ch = Channel.CreateBounded<T>(
            new BoundedChannelOptions(count) { FullMode = BoundedChannelFullMode.Wait });
        for (int i = 0; i < count; i++) ch.Writer.TryWrite(factory());
        return ch;
    }

    /// <summary>Average ms per stage from the last Convert run (for profiling output).</summary>
    public string? LastStageProfile { get; private set; }

    /// <summary>Per-phase breakdown of the depth stage from the last run (ms).</summary>
    public string? LastDepthProfile => _depth.LastProfile;

    private static double Ms(long ticks, long frames) => 1000.0 * ticks / System.Diagnostics.Stopwatch.Frequency / frames;

    /// <summary>Converts a single frame (used by preview mode).</summary>
    public PreviewResult ProcessPreviewFrame(byte[] rgb, int w, int h, ConversionSettings s)
    {
        var depth = _depth.EstimateDepth(rgb, w, h, w * 3);
        var synth = new StereoSynthesizer(s.DepthStrengthPercent, temporalSmoothing: 0f)
        {
            BackgroundFill = UseBackgroundFill,
            TrendFill = UseTrendFill,
        };
        var flat = new byte[w * h * 6];
        synth.Synthesize(rgb, depth, w, h, flat);
        if (s.Mode != StereoMode.Vr180)
            return new PreviewResult(flat, w * 2, h);

        int eyeSize = s.EyeSize > 0 ? s.EyeSize : Math.Clamp(EvenUp(h * 2), 640, 2160);
        var outBuf = new byte[eyeSize * 2 * eyeSize * 3];
        var proj = new EquirectProjector(s.ScreenFovDeg);
        proj.Project(flat, 0, w, h, w * 2 * 3, outBuf, 0, eyeSize * 2, eyeSize, eyeSize);
        proj.Project(flat, w, w, h, w * 2 * 3, outBuf, eyeSize, eyeSize * 2, eyeSize, eyeSize);
        return new PreviewResult(outBuf, eyeSize * 2, eyeSize);
    }

    private static int EvenUp(int v) => (v + 1) & ~1;

    /// <summary>
    /// Debug: estimate depth for one frame and return it alongside the synthesized right
    /// eye and hole mask, so artifacts can be inspected frame-by-frame.
    /// </summary>
    public (float[] depth, byte[] rightEye, bool[]? holes) DebugSynthesize(
        byte[] rgb, int w, int h, float strengthPercent, bool captureHoles = true)
    {
        float[] depth = _depth.EstimateDepth(rgb, w, h, w * 3);
        var synth = new StereoSynthesizer(strengthPercent) { CaptureHoles = captureHoles };
        var flat = new byte[w * h * 6];
        synth.Synthesize(rgb, depth, w, h, flat);

        // Extract right eye ROW BY ROW: in SBS buffer rows have stride w*6,
        // right eye starts at column w. A single contiguous BlockCopy would
        // interleave foreign rows (diagnostic-only bug, pipeline itself is fine).
        var rightEye = new byte[w * h * 3];
        int rowB = w * 3;
        for (int y = 0; y < h; y++)
            Buffer.BlockCopy(flat, y * rowB * 2 + rowB, rightEye, y * rowB, rowB);
        return (depth, rightEye, synth.LastHoleMask);
    }

    public void Dispose()
    {
        _gpu?.Dispose();
        _depth.Dispose();
    }
}
