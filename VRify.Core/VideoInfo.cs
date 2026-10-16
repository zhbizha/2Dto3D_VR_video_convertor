using System.Text.Json;

namespace VRify.Core;

public sealed record VideoInfo(int Width, int Height, double FrameRate, double DurationSec, long FrameCount, string? AudioCodec)
{
    /// <summary>Source path this info was probed from (set by Probe).</summary>
    public string? Path { get; init; }

    public static VideoInfo Probe(string path)
    {
        var json = FFmpegLocator.RunCaptured(FFmpegLocator.FfprobePath, new[]
        {
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames:format=duration",
            "-of", "json", path,
        });

        using var doc = JsonDocument.Parse(json);
        var stream = doc.RootElement.GetProperty("streams")[0];
        var w = stream.GetProperty("width").GetInt32();
        var h = stream.GetProperty("height").GetInt32();
        var parts = stream.GetProperty("r_frame_rate").GetString()!.Split('/');
        double fps = double.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture)
                   / double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);

        double duration = 0;
        if (stream.TryGetProperty("duration", out var d) && TryReadDouble(d, out var ds))
            duration = ds;
        else if (doc.RootElement.TryGetProperty("format", out var fmt) &&
                 fmt.TryGetProperty("duration", out var fd) && TryReadDouble(fd, out var fds))
            duration = fds;

        long frames = 0;
        if (stream.TryGetProperty("nb_frames", out var nf))
        {
            if (nf.ValueKind == JsonValueKind.Number && nf.TryGetInt64(out var nf64))
                frames = nf64;
            else if (nf.ValueKind == JsonValueKind.String && long.TryParse(nf.GetString(), out nf64))
                frames = nf64;
        }
        if (frames == 0 && duration > 0 && fps > 0)
            frames = (long)Math.Round(duration * fps);

        // Audio codec of the first audio stream (if any)
        string? audioCodec = null;
        var ajson = FFmpegLocator.RunCaptured(FFmpegLocator.FfprobePath, new[]
        {
            "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name", "-of", "csv=p=0", path,
        }).Trim();
        if (ajson.Length > 0) audioCodec = ajson;

        return new VideoInfo(w, h, fps, duration, frames, audioCodec) { Path = path };
    }

    /// <summary>ffprobe may emit numbers as JSON strings depending on version/muxer.</summary>
    private static bool TryReadDouble(JsonElement el, out double value)
    {
        if (el.ValueKind == JsonValueKind.Number) return el.TryGetDouble(out value);
        if (el.ValueKind == JsonValueKind.String)
            return double.TryParse(el.GetString(), System.Globalization.CultureInfo.InvariantCulture, out value);
        value = 0;
        return false;
    }
}
