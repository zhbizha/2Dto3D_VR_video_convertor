using System.Diagnostics;

namespace VRify.Core;

/// <summary>Streams decoded RGB24 frames out of ffmpeg (rawvideo over stdout).</summary>
public sealed class FrameReader : IDisposable
{
    private readonly Process _proc;
    private readonly Stream _stdout;
    private readonly StringWriter _stderr = new();
    public int Width { get; }
    public int Height { get; }
    public int FrameSize => Width * Height * 3;

    public FrameReader(string inputPath, int width, int height)
    {
        Width = width;
        Height = height;
        var psi = new ProcessStartInfo
        {
            FileName = FFmpegLocator.FfmpegPath,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        foreach (var a in new[]
        {
            "-v", "error",
            "-i", inputPath,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        })
            psi.ArgumentList.Add(a);

        _stderr = new StringWriter();
        _proc = Process.Start(psi) ?? throw new InvalidOperationException("Failed to start ffmpeg decoder");
        _proc.ErrorDataReceived += (_, e) => { if (e.Data != null) _stderr.WriteLine(e.Data); };
        _proc.BeginErrorReadLine();
        _stdout = _proc.StandardOutput.BaseStream;
    }

    /// <summary>Reads the next frame into <paramref name="buffer"/>. Returns bytes read (0 = EOF).</summary>
    public int ReadFrame(byte[] buffer)
    {
        int total = 0;
        while (total < FrameSize)
        {
            int n = _stdout.Read(buffer, total, FrameSize - total);
            if (n == 0) return total == 0 ? 0 : total; // partial tail shouldn't happen
            total += n;
        }
        return total;
    }

    public void Dispose()
    {
        try
        {
            _stdout.Dispose();
            if (!_proc.WaitForExit(3000)) _proc.Kill(entireProcessTree: true);
            else if (_proc.ExitCode != 0 && _proc.ExitCode != -1 && _proc.ExitCode != 255)
                throw new InvalidOperationException($"ffmpeg decoder failed (code {_proc.ExitCode}):\n{_stderr}");
        }
        catch { /* shutdown race */ }
        finally { _proc.Dispose(); }
    }
}
