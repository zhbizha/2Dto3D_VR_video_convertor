using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VRify.Core;

/// <summary>
/// Monocular depth estimation with Depth Anything V2 (ONNX) on GPU via DirectML.
/// Input: RGB frame. Output: relative depth map (higher = closer), normalized 0..1 per frame.
/// </summary>
public sealed class DepthEstimator : IDisposable
{
    // DPTImageProcessor defaults from preprocessor_config.json
    private const int TargetSize = 518;
    private const int MultipleOf = 14;
    private static readonly (float Mean, float Std)[] ChannelStats =
    {
        (0.485f, 0.229f), (0.456f, 0.224f), (0.406f, 0.225f),
    };

    private InferenceSession _session;
    private string _inputName;
    private string _outputName;
    private readonly string _modelPath;
    private readonly int _gpuDeviceId;
    private int _fixedW, _fixedH;                    // dimensions currently baked into _session
    private bool _dmlOk = true;
    private bool _dimsOk = true;
    private readonly SemaphoreSlim _gate = new(1, 1); // DirectML session is not re-entrant
    private readonly DepthRegularizer _regularizer = new(radius: 14, eps: 0.004f);
    private float[] _medianTmp = Array.Empty<float>();
    private float[] _depthOut = Array.Empty<float>();   // reused output buffer (no per-frame alloc)
    private readonly System.Diagnostics.Stopwatch _sw = new();

    /// <summary>When true, <see cref="LastProfile"/> is populated with per-phase timings.</summary>
    public static bool Profile { get; set; }

    /// <summary>
    /// When set, <see cref="EstimateDepth"/> and <see cref="RefineEdges"/> write the depth
    /// map at every stage to PNGs under this directory: depth_raw_small.png (after normalize
    /// + guided + median, before collapse), depth_collapse_small.png (after model-res
    /// collapse), depth_pre_refine.png (after bilinear upsample), depth_post_refine.png
    /// (after full-res collapse or guided). Used to localize where the silhouette steps
    /// are introduced.
    /// </summary>
    public static string? DebugDumpPath { get; set; }

    /// <summary>
    /// Pin the model's free height/width dimensions to the exact inference size. This lets ORT
    /// constant-fold the DPT head's data-dependent <c>Resize</c> sizes so DirectML (GPU) runs them
    /// instead of the CPU - roughly 6x faster inference. The trade-off is a sub-LSB difference
    /// (~0.008 on the normalized depth map) from DirectML's fp16 resize rounding. Set false to
    /// keep the dynamic-shape graph (CPU resize, bit-closer to the fp32 reference, much slower).
    /// </summary>
    public static bool PinModelDims { get; set; } = true;

    /// <summary>
    /// Collapse the network's smooth silhouette ramps (≈3 px at model resolution, which the
    /// 4x upsample turns into ≈12 px) into steps before the depth leaves this class. A ramp
    /// that wide makes the forward warp fold over / stretch across a whole band of pixels,
    /// which is the halo + rubber artifact on silhouettes. Gated by local "stepness" so
    /// genuine depth slopes are left bit-exact. Set false for A/B comparison.
    /// </summary>
    public static bool SharpenEdges { get; set; } = true;

    // Full-resolution ramp-collapse kernel. Mirrored by the HLSL CollapseRamps entry
    // point in GpuShaders.cs — keep the two in sync.
    public const int FullResKernelS = 12;
    public const int FullResKernelB = 26;
    public const float FullResGain = 5f;

    // Model-resolution ramp-collapse kernel (called from EstimateDepth on the
    // 994×742 model output). Tuned to actually compress the 30-40 px ramps the
    // DPT network emits on this clip's silhouettes — the old (kS=4, kB=12, gain=5)
    // only got them down to 31 px, which bilinear upsample then re-stretched to
    // 24 px on full-res and the warp happily laid clothing fragments over the
    // disocclusion band. (kS=6, kB=28, gain=14) compresses them to ≈6 px on
    // model-res → ≈2-3 px after upsample, where the warp's bilinear AA hides them.
    public const int ModelResKernelS = 6;
    public const int ModelResKernelB = 28;
    public const float ModelResGain = 10f;

    /// <summary>Per-phase breakdown of the last <see cref="EstimateDepth"/> call (ms).</summary>
    public string? LastProfile { get; private set; }

    /// <summary>Execution provider + shape-pinning status actually in use.</summary>
    public string Provider { get; private set; } = "unknown";

    public DepthEstimator(string modelPath, int gpuDeviceId = 0)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Depth model not found: {modelPath}", modelPath);

        _modelPath = modelPath;
        _gpuDeviceId = gpuDeviceId;

        // Probe with a session at the default size; the real video size rebuilds it
        // lazily in EstimateDepth (see EnsureSession).
        _session = BuildSession(TargetSize, TargetSize);
        _inputName = _session.InputMetadata.Keys.First();
        _outputName = _session.OutputMetadata.Keys.First();
    }

    /// <summary>
    /// Builds an inference session. The DPT head's <c>Resize</c> nodes have data-dependent
    /// output sizes, so with the model's dynamic height/width they are pinned to the CPU
    /// execution provider and dominate the runtime (~450 ms/frame at 4K vs ~70 ms).
    /// Freezing the free dimensions via AddFreeDimensionOverrideByName lets ORT constant-fold
    /// those sizes so DirectML can run them — a ~6x speedup. Falls back gracefully.
    /// </summary>
    private InferenceSession BuildSession(int rw, int rh)
    {
        var attempts = new List<(bool Dml, bool Pin)>();
        if (_dmlOk && _dimsOk && PinModelDims) attempts.Add((true, true));
        if (_dmlOk) attempts.Add((true, false));
        attempts.Add((false, false));

        Exception? last = null;
        foreach (var (dml, pin) in attempts)
        {
            try
            {
                var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
                if (dml) so.AppendExecutionProvider_DML(_gpuDeviceId);
                if (pin)
                {
                    so.AddFreeDimensionOverrideByName("batch_size", 1);
                    so.AddFreeDimensionOverrideByName("height", rh);
                    so.AddFreeDimensionOverrideByName("width", rw);
                }
                var session = new InferenceSession(_modelPath, so);
                _fixedW = pin ? rw : 0;
                _fixedH = pin ? rh : 0;
                Provider = (dml ? $"DirectML (GPU #{_gpuDeviceId})" : "CPU")
                           + (pin ? "" : " (dynamic dims)");
                return session;
            }
            catch (Exception ex)
            {
                last = ex;
                if (pin) _dimsOk = false;
                else if (dml) _dmlOk = false;
            }
        }
        throw new InvalidOperationException("Could not create an ONNX inference session.", last);
    }

    /// <summary>Rebuilds the session if the requested model input size changed.</summary>
    private void EnsureSession(int rw, int rh)
    {
        if (_session != null && _fixedW == rw && _fixedH == rh) return;
        _session?.Dispose();
        _session = BuildSession(rw, rh);
        _inputName = _session.InputMetadata.Keys.First();
        _outputName = _session.OutputMetadata.Keys.First();
    }

    /// <summary>
    /// Estimates depth for one RGB24 frame (stride = Width*3).
    /// Returns a float map of size Width*Height where 1 = closest, 0 = farthest.
    /// </summary>
    public float[] EstimateDepth(byte[] rgb, int width, int height, int stride)
    {
        _gate.Wait();
        try
        {
            // 1. Scale model input with source resolution. Fixed 518px was fine for ~1K
            //    sources (downsample ≈2×) but destroyed edges on 4K (downsample 7.6× →
            //    depth cannot resolve silhouettes → eroded corners, mushy edges).
            //    Target: never downsample more than ~4×; multiple of 14 (DPT requirement);
            //    cap 1540 to bound inference cost.
            int longest = Math.Max(width, height);
            int targetLong = (int)Math.Ceiling(longest / 4.0 / MultipleOf) * MultipleOf;
            targetLong = Math.Clamp(targetLong, TargetSize, 1540);
            double scale = (double)targetLong / longest;
            int rw = Math.Max(MultipleOf, (int)Math.Round(width * scale / MultipleOf) * MultipleOf);
            int rh = Math.Max(MultipleOf, (int)Math.Round(height * scale / MultipleOf) * MultipleOf);

            EnsureSession(rw, rh);   // pin model dims so DML (not CPU) runs the head's Resize

            var resized = new byte[rw * rh * 3];
            _sw.Restart();
            BilinearResizeRgb(rgb, width, height, resized, rw, rh);
            double tResize = _sw.Elapsed.TotalMilliseconds;

            // 2. CHW float tensor, normalized with ImageNet stats.
            _sw.Restart();
            var tensor = new DenseTensor<float>(new[] { 1, 3, rh, rw });
            float inv = 1f / 255f;
            int plane = rh * rw;
            var tSpan = tensor.Buffer.Span;
            for (int c = 0; c < 3; c++)
            {
                var (m, s) = ChannelStats[c];
                var dstC = tSpan.Slice(c * plane, plane);
                for (int i = 0, p = c; i < plane; i++, p += 3)
                    dstC[i] = (resized[p] * inv - m) / s;
            }
            double tTensor = _sw.Elapsed.TotalMilliseconds;

            // 3. Inference.
            _sw.Restart();
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
            using var results = _session.Run(inputs);
            double tInfer = _sw.Elapsed.TotalMilliseconds;

            _sw.Restart();
            var depthSmall = results.First().AsEnumerable<float>().ToArray(); // rh*rw
            double tExtract = _sw.Elapsed.TotalMilliseconds;

            // 4. Filters run at MODEL resolution (15× cheaper than full res at 4K):
            //    normalize → guided flatten → median. Then plain bilinear upsample.
            //    Edge sharpness is restored later by RefineEdges (full-res guided
            //    re-snap), so the old joint-bilateral upsample (585M exp() calls/frame
            //    at 4K) is gone entirely.
            _sw.Restart();
            NormalizeInPlace(depthSmall);
            _regularizer.Apply(depthSmall, resized, rw, rh, rw * 3);
            MedianFilter3x3(depthSmall, rw, rh);
            if (DebugDumpPath is { } dump) DumpDepth(dump, "depth_raw_small", depthSmall, rw, rh);
            if (SharpenEdges) CollapseRamps(depthSmall, rw, rh, kS: ModelResKernelS, kB: ModelResKernelB, gain: ModelResGain);
            if (DebugDumpPath is { } dump2) DumpDepth(dump2, "depth_collapse_small", depthSmall, rw, rh);
            double tFilter = _sw.Elapsed.TotalMilliseconds;

            _sw.Restart();
            var depth = _depthOut.Length == width * height ? _depthOut : _depthOut = new float[width * height];
            BilinearResizeFloat(depthSmall, rw, rh, depth, width, height);
            if (DebugDumpPath is { } dump3) DumpDepth(dump3, "depth_pre_refine", depth, width, height);
            double tUp = _sw.Elapsed.TotalMilliseconds;

            if (Profile)
                LastProfile = $"res={rw}x{rh} resize={tResize:F0} tensor={tTensor:F0} infer={tInfer:F0} " +
                             $"extract={tExtract:F0} filt={tFilter:F0} up={tUp:F0}";
            return depth;
        }
        finally { _gate.Release(); }
    }

    /// <summary>
    /// Collapse smooth depth ramps into steps, per row (disparity is horizontal-only).
    /// stepness = range(±kS) / range(±kB) is ≈1 at a real discontinuity and ≈kS/kB on a
    /// uniform slope, so the local contrast stretch is blended in only where the structure
    /// actually is a step — verified to leave smooth regions bit-exact.
    /// Also mirrored by the HLSL CollapseRamps entry point in GpuShaders.cs.
    /// </summary>
    public static void CollapseRamps(float[] d, int w, int h, int kS, int kB, float gain)
    {
        var src = (float[])d.Clone();   // read the original, never own output (no feedback)
        Parallel.For(0, h, y =>
        {
            int row = y * w;
            for (int x = 0; x < w; x++)
            {
                float loS = float.MaxValue, hiS = float.MinValue, loB = float.MaxValue, hiB = float.MinValue;
                int s0 = Math.Max(0, x - kS), s1 = Math.Min(w - 1, x + kS);
                int b0 = Math.Max(0, x - kB), b1 = Math.Min(w - 1, x + kB);
                for (int k = s0; k <= s1; k++) { float v = src[row + k]; if (v < loS) loS = v; if (v > hiS) hiS = v; }
                for (int k = b0; k <= b1; k++) { float v = src[row + k]; if (v < loB) loB = v; if (v > hiB) hiB = v; }

                float rS = hiS - loS, rB = hiB - loB;
                float stepness = rB > 1e-6f ? rS / rB : 0f;
                float t = Math.Clamp((stepness - 0.5f) / 0.3f, 0f, 1f);
                float wt = t * t * (3f - 2f * t);

                float v0 = src[row + x];
                if (wt <= 0f || rS < 0.02f) { d[row + x] = v0; continue; }

                float tt = (v0 - loS) / rS;
                float tc = Math.Clamp((tt - 0.5f) * gain + 0.5f, 0f, 1f);
                d[row + x] = v0 + (loS + tc * rS - v0) * wt;
            }
        });
    }

    /// <summary>
    /// Post-temporal edge re-snap. Temporal EMA widens depth transitions where the
    /// silhouette moved between frames, so the transition is pulled back onto the current
    /// frame.
    ///
    /// This used to be a full-res guided filter (radius 14). Measured on a 4K silhouette:
    /// that refine WIDENED the transition (13 -> 14 px, p90 18 -> 24 px) and nudged genuine
    /// depth slopes (0.00057 mean change in smooth regions), i.e. it worked against edge
    /// sharpness — the opposite of its stated purpose. It now runs the same gated ramp
    /// collapse used at model resolution, which measured 13 -> 4 px with a 0.00000 change
    /// in smooth regions.
    /// </summary>
    public void RefineEdges(float[] depth, byte[] rgb, int width, int height, int stride)
    {
        _gate.Wait();
        try
        {
            if (DebugDumpPath is { } dump0) DumpDepth(dump0, "depth_refine_in", depth, width, height);
            if (SharpenEdges)
            {
                CollapseRamps(depth, width, height, kS: FullResKernelS, kB: FullResKernelB, gain: FullResGain);
                HardStepPass(depth, width, height, kD: 3, kB: 10, slope: 0.12f);
            }
            else _regularizer.Apply(depth, rgb, width, height, stride);
            MedianFilter3x3(depth, width, height);
            if (DebugDumpPath is { } dump1) DumpDepth(dump1, "depth_refine_out", depth, width, height);
        }
        finally { _gate.Release(); }
    }

    /// <summary>
    /// Hard-step pass: after the gated ramp collapse, walk each row and wherever the
    /// local range in a wide window is concentrated in a small inner window (stepness
    /// ratio high), snap the wide window to its two poles. This is what finally turns
    /// the last 4-8 px of bilinear-upsample AA into a true step on the warp input —
    /// without it, the warp lays clothing fragments over the disocclusion band on the
    /// right eye (the "horizontal stair" the user kept seeing on the dark-cliff fill).
    ///
    /// The stepness gate (range(±kD)/range(±kB)) is the same one CollapseRamps uses,
    /// so the pass only fires on actual discontinuities and leaves soft slopes
    /// (curtains, hair, soft fabrics) bit-exact.
    /// </summary>
    public static void HardStepPass(float[] d, int w, int h, int kD, int kB, float slope)
    {
        var src = (float[])d.Clone();
        Parallel.For(0, h, y =>
        {
            int row = y * w;
            for (int x = 0; x < w; x++)
            {
                int s0 = Math.Max(0, x - kD), s1 = Math.Min(w - 1, x + kD);
                int b0 = Math.Max(0, x - kB), b1 = Math.Min(w - 1, x + kB);
                float mnS = float.MaxValue, mxS = float.MinValue, mnB = float.MaxValue, mxB = float.MinValue;
                for (int k = s0; k <= s1; k++) { var v = src[row + k]; if (v < mnS) mnS = v; if (v > mxS) mxS = v; }
                if (mxS - mnS <= slope) continue;
                for (int k = b0; k <= b1; k++) { var v = src[row + k]; if (v < mnB) mnB = v; if (v > mxB) mxB = v; }
                float rB = mxB - mnB;
                if (rB < 1e-6f) continue;
                float stepness = (mxS - mnS) / rB;
                if (stepness < 0.75f) continue;       // very sharp edge only — avoids shattering soft gradients
                // Count: how many pixels in the wide window are clearly on each side of (mnB+mxB)/2?
                float mid = (mnB + mxB) * 0.5f;
                int nLow = 0, nHigh = 0;
                for (int k = b0; k <= b1; k++) { if (src[row + k] < mid) nLow++; else nHigh++; }
                if (Math.Min(nLow, nHigh) < 2) continue;   // both poles must have at least 2 supporting pixels
                d[row + x] = mid >= src[row + x] ? mxB : mnB;
            }
        });
    }

    /// <summary>3×3 median filter: removes single-pixel depth spikes without blurring edges. Row-parallel, grow-only buffer.</summary>
    private void MedianFilter3x3(float[] map, int w, int h)
    {
        if (_medianTmp.Length < map.Length) _medianTmp = new float[map.Length];
        var tmp = _medianTmp;
        Parallel.For(0, h, y =>
        {
            int y0 = y > 0 ? y - 1 : 0, y1 = y < h - 1 ? y + 1 : h - 1;
            Span<float> win = stackalloc float[9];
            int outRow = y * w;
            for (int x = 0; x < w; x++)
            {
                int x0 = x > 0 ? x - 1 : 0, x1 = x < w - 1 ? x + 1 : w - 1;
                int k = 0;
                for (int yy = y0; yy <= y1; yy++)
                    for (int xx = x0; xx <= x1; xx++)
                        win[k++] = map[yy * w + xx];
                // insertion sort of ≤9 elements
                for (int i = 1; i < k; i++)
                {
                    float v = win[i];
                    int j = i - 1;
                    while (j >= 0 && win[j] > v) { win[j + 1] = win[j]; j--; }
                    win[j + 1] = v;
                }
                tmp[outRow + x] = win[k / 2];
            }
        });
        Array.Copy(tmp, map, map.Length);
    }

    private static void NormalizeInPlace(float[] map)
    {
        float min = float.MaxValue, max = float.MinValue;
        foreach (var v in map)
        {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        float range = max - min;
        if (range < 1e-6f) { Array.Clear(map); return; }
        // Depth Anything: bigger = closer -> invert so 1 = closest
        for (int i = 0; i < map.Length; i++)
            map[i] = (map[i] - min) / range;
    }

    private static void DumpDepth(string dir, string name, float[] depth, int w, int h)
    {
        System.IO.Directory.CreateDirectory(dir);
        var bytes = new byte[depth.Length * 4];
        System.Buffer.BlockCopy(depth, 0, bytes, 0, bytes.Length);
        System.IO.File.WriteAllBytes(System.IO.Path.Combine(dir, name + ".bin"), bytes);
        System.IO.File.WriteAllText(System.IO.Path.Combine(dir, name + ".txt"), $"{w} {h}");
    }

    private static void BilinearResizeRgb(byte[] src, int sw, int sh, byte[] dst, int dw, int dh)
    {
        float sx = (float)sw / dw, sy = (float)sh / dh;
        Parallel.For(0, dh, y =>
        {
            float fy = Math.Clamp((y + 0.5f) * sy - 0.5f, 0f, sh - 1f);
            int y0 = (int)fy, y1 = Math.Min(y0 + 1, sh - 1);
            float ry = fy - y0;
            for (int x = 0; x < dw; x++)
            {
                float fx = Math.Clamp((x + 0.5f) * sx - 0.5f, 0f, sw - 1f);
                int x0 = (int)fx, x1 = Math.Min(x0 + 1, sw - 1);
                float rx = fx - x0;
                int d = (y * dw + x) * 3;
                for (int c = 0; c < 3; c++)
                {
                    float v00 = src[(y0 * sw + x0) * 3 + c];
                    float v01 = src[(y0 * sw + x1) * 3 + c];
                    float v10 = src[(y1 * sw + x0) * 3 + c];
                    float v11 = src[(y1 * sw + x1) * 3 + c];
                    float v = v00 * (1 - rx) * (1 - ry) + v01 * rx * (1 - ry)
                             + v10 * (1 - rx) * ry + v11 * rx * ry;
                    dst[d + c] = (byte)Math.Clamp(v, 0, 255);
                }
            }
        });
    }

    private static void BilinearResizeFloat(float[] src, int sw, int sh, float[] dst, int dw, int dh)
    {
        float sx = (float)sw / dw, sy = (float)sh / dh;
        Parallel.For(0, dh, y =>
        {
            float fy = Math.Clamp((y + 0.5f) * sy - 0.5f, 0f, sh - 1f);
            int y0 = (int)fy, y1 = Math.Min(y0 + 1, sh - 1);
            float ry = fy - y0;
            for (int x = 0; x < dw; x++)
            {
                float fx = Math.Clamp((x + 0.5f) * sx - 0.5f, 0f, sw - 1f);
                int x0 = (int)fx, x1 = Math.Min(x0 + 1, sw - 1);
                float rx = fx - x0;
                float v00 = src[y0 * sw + x0];
                float v01 = src[y0 * sw + x1];
                float v10 = src[y1 * sw + x0];
                float v11 = src[y1 * sw + x1];
                dst[y * dw + x] = v00 * (1 - rx) * (1 - ry) + v01 * rx * (1 - ry)
                                + v10 * (1 - rx) * ry + v11 * rx * ry;
            }
        });
    }

    public void Dispose() { _session.Dispose(); _gate.Dispose(); }
}
