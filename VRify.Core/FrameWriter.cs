using System.Diagnostics;

namespace VRify.Core;

/// <summary>Feeds raw RGB24/RGBA SBS frames to ffmpeg and encodes H.264/H.265 MP4, muxing source audio.
/// Uses NVENC (GPU hardware encoder) when available; falls back to libx265/libx264.</summary>
public sealed class FrameWriter : IDisposable
{
    private readonly Process _proc;
    private readonly Stream _stdin;
    private readonly StringWriter _stderr = new();

    public int Width { get; }
    public int Height { get; }
    public int FrameSize => Width * Height * (InputRgba ? 4 : 3);
    public bool InputRgba { get; }

    private static int _nvencHevc = -1, _nvencH264 = -1;   // -1 = unknown, 0 = no, 1 = yes

    /// <summary>Probes ffmpeg once for NVENC encoder availability (cached for the process lifetime).</summary>
    private static bool HasNvenc(bool hevc)
    {
        ref int cache = ref (hevc ? ref _nvencHevc : ref _nvencH264);
        int v = Volatile.Read(ref cache);
        if (v < 0)
        {
            v = Probe(hevc ? "hevc_nvenc" : "h264_nvenc") ? 1 : 0;
            Volatile.Write(ref cache, v);
        }
        return v == 1;

        static bool Probe(string name)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = FFmpegLocator.FfmpegPath,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                };
                psi.ArgumentList.Add("-hide_banner");
                psi.ArgumentList.Add("-encoders");
                using var p = Process.Start(psi);
                if (p == null) return false;
                string output = p.StandardOutput.ReadToEnd();
                p.StandardError.ReadToEnd();
                p.WaitForExit(10000);
                return output.Contains(name);
            }
            catch { return false; }
        }
    }

    public FrameWriter(string outputPath, VideoInfo sourceInfo, int width, int height, double fps,
        int crf = 20, bool hevc = true, int? maxBitrateKbps = null, bool inputRgba = false)
    {
        Width = width;
        Height = height;
        InputRgba = inputRgba;

        var args = new List<string>
        {
            "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", inputRgba ? "rgba" : "rgb24",
            "-s", $"{width}x{height}",
            "-r", fps.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "-i", "pipe:0",
        };
        bool hasAudio = !string.IsNullOrEmpty(sourceInfo.AudioCodec);
        if (hasAudio)
        {
            args.AddRange(new[] { "-i", sourceInfo.Path!, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy", "-shortest" });
        }
        else
        {
            args.AddRange(new[] { "-map", "0:v:0" });
        }

        // NVENC hardware encode (GPU) with software fallback
        bool useNvenc = HasNvenc(hevc);
        if (useNvenc)
        {
            args.AddRange(new[]
            {
                "-c:v", hevc ? "hevc_nvenc" : "h264_nvenc",
                "-preset", "p4",
                "-rc", "vbr",
                "-cq", Math.Clamp(crf + 3, 1, 51).ToString(),
                "-b:v", "0",
            });
        }
        else
        {
            args.AddRange(new[]
            {
                "-c:v", hevc ? "libx265" : "libx264",
                "-preset", "fast",
                "-crf", crf.ToString(),
            });
        }
        args.AddRange(new[]
        {
            "-pix_fmt", "yuv420p",
            "-tag:v", "hvc1", // QuickView/Apple compatibility for HEVC
        });
        if (maxBitrateKbps is int kb)
            args.AddRange(new[] { "-b:v", $"{kb}k", "-maxrate", $"{kb}k", "-bufsize", $"{kb * 2}k" });
        args.Add(outputPath);

        var psiEnc = new ProcessStartInfo
        {
            FileName = FFmpegLocator.FfmpegPath,
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        foreach (var a in args) psiEnc.ArgumentList.Add(a);

        _proc = Process.Start(psiEnc) ?? throw new InvalidOperationException("Failed to start ffmpeg encoder");
        _proc.ErrorDataReceived += (_, e) => { if (e.Data != null) _stderr.WriteLine(e.Data); };
        _proc.BeginErrorReadLine();
        _stdin = _proc.StandardInput.BaseStream;
    }

    public void WriteFrame(byte[] buffer) => _stdin.Write(buffer, 0, FrameSize);

    public void Dispose()
    {
        try
        {
            _stdin.Dispose();
            if (!_proc.WaitForExit(30000)) _proc.Kill(entireProcessTree: true);
            else if (_proc.ExitCode != 0)
                throw new InvalidOperationException($"ffmpeg encoder failed (code {_proc.ExitCode}):\n{_stderr}");
        }
        catch (InvalidOperationException) { throw; } // surface encoder errors
        catch { /* shutdown race */ }
        finally { _proc.Dispose(); }
    }
}
