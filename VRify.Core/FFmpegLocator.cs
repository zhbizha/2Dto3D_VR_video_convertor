using System.Diagnostics;

namespace VRify.Core;

/// <summary>Locates ffmpeg/ffprobe executables: PATH, common install dirs, winget.</summary>
public static class FFmpegLocator
{
    private static readonly Lazy<string?> Ffmpeg = new(() => Find("ffmpeg.exe"));
    private static readonly Lazy<string?> Ffprobe = new(() => Find("ffprobe.exe"));

    public static string FfmpegPath => Ffmpeg.Value ?? throw new FileNotFoundException(
        "ffmpeg.exe not found. Install it (e.g. 'winget install Gyan.FFmpeg') or put it next to VRify.exe.");

    public static string FfprobePath => Ffprobe.Value ?? throw new FileNotFoundException(
        "ffprobe.exe not found. Install it (e.g. 'winget install Gyan.FFmpeg').");

    private static string? Find(string name)
    {
        // 1. Next to the app (portable layout)
        var beside = Path.Combine(AppContext.BaseDirectory, name);
        if (File.Exists(beside)) return beside;

        // 2. PATH (refreshed because it may have been set by the installer after process start)
        var pathEnv = Environment.GetEnvironmentVariable("PATH") ?? "";
        foreach (var dir in pathEnv.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            try
            {
                var candidate = Path.Combine(dir.Trim(), name);
                if (File.Exists(candidate)) return candidate;
            }
            catch { /* invalid PATH entry */ }
        }

        // 3. Winget default install location (Gyan.FFmpeg)
        var wingetRoot = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Microsoft", "WinGet", "Packages");
        if (Directory.Exists(wingetRoot))
        {
            foreach (var pkg in Directory.EnumerateDirectories(wingetRoot, "Gyan.FFmpeg*"))
            {
                var found = Directory.EnumerateFiles(pkg, name, SearchOption.AllDirectories).FirstOrDefault();
                if (found != null) return found;
            }
        }

        // 4. Well-known manual installs
        foreach (var root in new[] { @"C:\ffmpeg", @"C:\Program Files\ffmpeg", @"D:\ffmpeg" })
        {
            var candidate = Path.Combine(root, "bin", name);
            if (File.Exists(candidate)) return candidate;
        }

        return null;
    }

    /// <summary>Runs a process to completion and returns stdout (used for ffprobe).</summary>
    internal static string RunCaptured(string exe, IReadOnlyList<string> args)
    {
        var psi = new ProcessStartInfo
        {
            FileName = exe,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        foreach (var a in args) psi.ArgumentList.Add(a);

        using var p = Process.Start(psi) ?? throw new InvalidOperationException($"Failed to start {exe}");
        var stdout = p.StandardOutput.ReadToEnd();
        var stderr = p.StandardError.ReadToEnd();
        p.WaitForExit();
        if (p.ExitCode != 0)
            throw new InvalidOperationException($"{exe} exited with code {p.ExitCode}:\n{stderr}");
        return stdout;
    }
}
