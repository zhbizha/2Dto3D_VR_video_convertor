using VRify.Core;

// VRify CLI
// Usage:
//   VRify.Cli <input> [output] [strength% 0.5-6] [smoothing 0-0.9] [flat|180] [fov]
//   VRify.Cli debug <input> <frameA> <frameB> <strength%> <outdir>   — dump depth/hole maps
//   VRify.Cli ramptest <dir> [y0 y1] [x0 x1]                          — measure depth ramp widths
//   VRify.Cli settings [--set key=value ...] [--reset]                 — show/edit saved presets

if (args.Length >= 1 && args[0].Equals("debug", StringComparison.OrdinalIgnoreCase))
    return RunDebug(args);

if (args.Length >= 1 && args[0].Equals("ramptest", StringComparison.OrdinalIgnoreCase))
    return RunRampTest(args);

if (args.Length >= 1 && args[0].Equals("settings", StringComparison.OrdinalIgnoreCase))
    return RunSettings(args);

if (args.Length < 1)
{
    Console.Error.WriteLine("Usage: VRify.Cli <input> [output] [strength%] [smoothing] [flat|180] [fov]");
    Console.Error.WriteLine("       VRify.Cli debug <input> <frameA> <frameB> <strength%> <outdir>");
    Console.Error.WriteLine("       VRify.Cli ramptest <dir> [y0 y1] [x0 x1]");
    Console.Error.WriteLine("       VRify.Cli settings [--set fov=110 strength=2.5 smoothing=0 hevc=true mode=180] [--reset]");
    return 1;
}

string input = args[0];
string output = args.Length > 1 ? args[1] : Path.ChangeExtension(input, null) + "_VR_SBS180.mp4";
float strength = args.Length > 2 ? float.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture) : 2.5f;
float smoothing = args.Length > 3 ? float.Parse(args[3], System.Globalization.CultureInfo.InvariantCulture) : 0.6f;
var mode = args.Length > 4 && args[4].Equals("180", StringComparison.OrdinalIgnoreCase)
    ? StereoMode.Vr180 : StereoMode.FlatSbs;
double fov = args.Length > 5
    ? double.Parse(args[5], System.Globalization.CultureInfo.InvariantCulture) : 140.0;

Console.WriteLine($"Input : {input}");
var info = VideoInfo.Probe(input);
Console.WriteLine($"Probe : {info.Width}x{info.Height} @ {info.FrameRate:0.##} fps, ~{info.FrameCount} frames");
Console.WriteLine($"Mode  : {(mode == StereoMode.Vr180 ? $"VR180 equirect (screen {fov:0}°)" : "flat SBS")}");

using var conv = new VrConverter(VrConverter.FindModel());
Console.WriteLine($"Model : {conv.Provider} ({Path.GetFileName(VrConverter.FindModel())})");
Console.WriteLine($"GPU   : {conv.GpuInfo ?? "CPU (no D3D11 device)"}");

var progress = new Progress<ConversionProgress>(p =>
{
    Console.Write($"\rFrame {p.FramesDone}/{p.FramesTotal} ({p.Percent:0.0}%) {p.Fps:0.0} fps   ");
});

var sw = System.Diagnostics.Stopwatch.StartNew();
DepthEstimator.Profile = true;
conv.Convert(new ConversionSettings(input, output, strength, smoothing,
    Mode: mode, ScreenFovDeg: fov), progress);
Console.WriteLine();
if (conv.LastStageProfile != null)
    Console.WriteLine($"Profile: {conv.LastStageProfile}");
if (conv.LastDepthProfile != null)
    Console.WriteLine($"Depth  : {conv.LastDepthProfile}");
Console.WriteLine($"Done in {sw.Elapsed.TotalSeconds:0.0}s -> {output}");
Console.WriteLine($"Size  : {new FileInfo(output).Length / 1024 / 1024.0:0.0} MB");
return 0;

// ---------------------------------------------------------------------------

static int RunDebug(string[] args)
{
    if (args.Length < 6)
    {
        Console.Error.WriteLine("Usage: VRify.Cli debug <input> <frameA> <frameB> <strength%> <outdir>");
        return 1;
    }
    string input = args[1];
    int fa = int.Parse(args[2]), fb = int.Parse(args[3]);
    float strength = float.Parse(args[4], System.Globalization.CultureInfo.InvariantCulture);
    string outDir = args[5];
    Directory.CreateDirectory(outDir);

    var info = VideoInfo.Probe(input);
    using var conv = new VrConverter(VrConverter.FindModel());
    Console.WriteLine($"Model : {conv.Provider}");

    // Decode up to frame fb
    var frames = new Dictionary<int, byte[]>();
    using (var reader = new FrameReader(input, info.Width, info.Height))
    {
        var buf = new byte[info.Width * info.Height * 3];
        for (int i = 0; i <= Math.Max(fa, fb); i++)
        {
            if (reader.ReadFrame(buf) != buf.Length) break;
            if (i == fa || i == fb) frames[i] = (byte[])buf.Clone();
        }
    }
    if (!frames.ContainsKey(fa) || !frames.ContainsKey(fb))
    {
        Console.Error.WriteLine("Frame index out of range.");
        return 1;
    }

    int w = info.Width, h = info.Height;
    var synth = new StereoSynthesizer(strength);

    // Per-stage depth dump so we can localize where the silhouette steps come from.
    var depthDump = Path.Combine(outDir, "depth_stages");
    DepthEstimator.DebugDumpPath = depthDump;
    if (Directory.Exists(depthDump)) Directory.Delete(depthDump, recursive: true);

    // Depth of both frames
    var (dA, rightA, holesA) = conv.DebugSynthesize(frames[fa], w, h, strength);
    var (dB, rightB, holesB) = conv.DebugSynthesize(frames[fb], w, h, strength);

    // Also dump full-res RefineEdges input/output on frame A (EstimateDepth returns
    // the pre-refine depth; we run RefineEdges once on a clone to get the production
    // post-refine state for the same frame).
    var dA_refineIn = (float[])dA.Clone();
    conv.Depth.RefineEdges(dA_refineIn, frames[fa], w, h, w * 3);
    WriteGray(Path.Combine(outDir, $"depth_f{fa}_refined.png"), dA_refineIn, w, h);
    WriteGray(Path.Combine(outDir, $"depth_f{fa}_prerefine.png"), dA, w, h);

    // Stage-by-stage .bin → PNG (window: 200x120 at full-res around x=2262,y=830).
    DumpDepthBinAsPng(Path.Combine(depthDump, "depth_raw_small.bin"),      Path.Combine(depthDump, "stage1_raw_small.png"),      567, 209, 50, 30, 12);
    DumpDepthBinAsPng(Path.Combine(depthDump, "depth_collapse_small.bin"), Path.Combine(depthDump, "stage2_collapse_small.png"), 567, 209, 50, 30, 12);
    DumpDepthBinAsPng(Path.Combine(depthDump, "depth_pre_refine.bin"),     Path.Combine(depthDump, "stage3_pre_refine.png"),     2262, 830, 200, 120, 3);
    DumpDepthBinAsPng(Path.Combine(depthDump, "depth_refine_in.bin"),      Path.Combine(depthDump, "stage4_refine_in.png"),      2262, 830, 200, 120, 3);
    DumpDepthBinAsPng(Path.Combine(depthDump, "depth_refine_out.bin"),     Path.Combine(depthDump, "stage5_refine_out.png"),     2262, 830, 200, 120, 3);
    Console.WriteLine($"Stage PNGs in {depthDump}");

    // 1. Depth maps (as grayscale; high = near)
    WriteGray(Path.Combine(outDir, $"depth_f{fa}.png"), dA, w, h);
    WriteGray(Path.Combine(outDir, $"depth_f{fb}.png"), dB, w, h);

    // 2. Temporal depth difference — flicker magnitude (this is what "noisy artifacts" look like)
    var diff = new float[w * h];
    float maxD = 0f;
    for (int i = 0; i < diff.Length; i++)
    {
        diff[i] = MathF.Abs(dA[i] - dB[i]);
        if (diff[i] > maxD) maxD = diff[i];
    }
    Console.WriteLine($"Depth diff between frames {fa}/{fb}: max={maxD:F3}, mean={diff.Average():F3}");
    WriteGray(Path.Combine(outDir, $"depthdiff_f{fa}_f{fb}.png"), diff, w, h, normalize: true);

    // 3. Right eyes
    WriteRgb(Path.Combine(outDir, $"righteye_f{fa}.png"), rightA, w, h);
    WriteRgb(Path.Combine(outDir, $"righteye_f{fb}.png"), rightB, w, h);

    // 4. Hole masks (white = hole = inpainted region)
    if (holesA != null)
    {
        var holeGray = holesA.Select(b => b ? 1f : 0f).ToArray();
        WriteGray(Path.Combine(outDir, $"holes_f{fa}.png"), holeGray, w, h, normalize: false);
        Console.WriteLine($"Holes: {holesA.Count(x => x)} px ({100.0 * holesA.Count(x => x) / (w * h):0.00}%)");
    }
    if (holesB != null)
        Console.WriteLine($"Holes: {holesB.Count(x => x)} px ({100.0 * holesB.Count(x => x) / (w * h):0.00}%)");

    // 5. Source frames for reference
    WriteRgb(Path.Combine(outDir, $"source_f{fa}.png"), frames[fa], w, h);
    WriteRgb(Path.Combine(outDir, $"source_f{fb}.png"), frames[fb], w, h);

    Console.WriteLine($"Dumped to {outDir}");
    return 0;
}

static int RunRampTest(string[] args)
{
    if (args.Length < 2) { Console.Error.WriteLine("Usage: VRify.Cli ramptest <dir> [y0 y1] [x0 x1]"); return 1; }
    string dir = args[1];
    int y0 = args.Length > 2 ? int.Parse(args[2]) : 830;
    int y1 = args.Length > 3 ? int.Parse(args[3]) : 950;
    int x0 = args.Length > 4 ? int.Parse(args[4]) : 2300;
    int x1 = args.Length > 5 ? int.Parse(args[5]) : 2400;

    string[] stages = { "depth_raw_small", "depth_collapse_small", "depth_pre_refine", "depth_refine_in", "depth_refine_out" };
    Console.WriteLine($"  stage                     mean   p50   p90   max   (native px, *4 = full-res equivalent)");
    foreach (var st in stages)
    {
        string bin = Path.Combine(dir, st + ".bin");
        string txt = Path.Combine(dir, st + ".txt");
        if (!File.Exists(bin) || !File.Exists(txt)) { Console.WriteLine($"  {st,-24}  (no .bin)"); continue; }
        var dims = File.ReadAllText(txt).Trim().Split(' ');
        int W = int.Parse(dims[0]), H = int.Parse(dims[1]);
        var raw = File.ReadAllBytes(bin);
        var vals = new float[W * H];
        Buffer.BlockCopy(raw, 0, vals, 0, raw.Length);

        var widths = new System.Collections.Generic.List<int>();
        for (int y = y0; y < y1 && y < H; y++)
        {
            // Edge = transition inside [x0, x1]: find leftmost x where vals < threshold
            // (going right). The first dip below 0.3 inside the window is the silhouette.
            int ex = -1;
            for (int x = x0; x < x1 && x < W; x++) { if (vals[y * W + x] < 0.3f) { ex = x; break; } }
            if (ex < 0) continue;
            int lo = Math.Max(0, ex - 20), hi = Math.Min(W - 1, ex + 20);
            float mn = float.MaxValue, mx = float.MinValue;
            for (int x = lo; x <= hi; x++) { var v = vals[y * W + x]; if (v < mn) mn = v; if (v > mx) mx = v; }
            if (mx - mn < 0.01f) continue;
            float t10 = mn + 0.1f * (mx - mn);
            float t90 = mn + 0.9f * (mx - mn);
            int x10 = ex, x90 = ex;
            for (int x = ex; x >= lo; x--) { if (vals[y * W + x] <= t10) { x10 = x; break; } x10 = x; }
            for (int x = ex; x <= hi; x++) { if (vals[y * W + x] >= t90) { x90 = x; break; } x90 = x; }
            widths.Add(Math.Abs(x90 - x10));
        }
        if (widths.Count == 0) { Console.WriteLine($"  {st,-24}  (no edge in y={y0}..{y1})"); continue; }
        var sorted = widths.OrderBy(w => w).ToArray();
        double mean = widths.Average();
        int p50 = sorted[sorted.Length / 2];
        int p90 = sorted[(int)(sorted.Length * 0.9)];
        int max = sorted.Max();
        int scale = (st == "depth_raw_small" || st == "depth_collapse_small") ? 4 : 1;
        Console.WriteLine($"  {st,-24}  {mean,5:0.0}  {p50,3}  {p90,3}  {max,3}   (full-res eq: {mean*scale:0.0})");
    }
    return 0;
}

static int RunSettings(string[] args)
{
    var s = UserSettings.Load();
    bool reset = args.Any(a => a.Equals("--reset", StringComparison.OrdinalIgnoreCase));
    if (reset) s = new UserSettings();

    for (int i = 1; i < args.Length; i++)
    {
        var a = args[i];
        if (!a.Contains('=')) continue;
        var kv = a.Split('=', 2);
        var key = kv[0].TrimStart('-', '/').ToLowerInvariant();
        var val = kv[1];
        try
        {
            switch (key)
            {
                case "fov" or "screenfov" or "screen":
                    s.ScreenFovDeg = double.Parse(val, System.Globalization.CultureInfo.InvariantCulture);
                    break;
                case "strength" or "depth":
                    s.DepthStrengthPercent = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture);
                    break;
                case "smoothing" or "smooth":
                    s.TemporalSmoothing = float.Parse(val, System.Globalization.CultureInfo.InvariantCulture);
                    break;
                case "hevc":
                    s.Hevc = val.Trim().ToLowerInvariant() is "1" or "true" or "yes" or "on";
                    break;
                case "mode":
                    s.Mode = val.Trim().ToLowerInvariant() is "180" or "vr180" or "equirect"
                        ? StereoMode.Vr180 : StereoMode.FlatSbs;
                    break;
                default:
                    Console.Error.WriteLine($"Unknown setting '{key}' (fov, strength, smoothing, hevc, mode).");
                    break;
            }
        }
        catch (FormatException)
        {
            Console.Error.WriteLine($"Bad value for '{key}': '{val}'");
            return 1;
        }
    }

    s.Normalize();
    s.Save();
    Console.WriteLine($"File  : {UserSettings.FilePath}");
    Console.WriteLine($"Value : strength={s.DepthStrengthPercent:0.0}%  smoothing={s.TemporalSmoothing:0.0}  " +
                      $"hevc={s.Hevc}  mode={s.Mode}  fov={s.ScreenFovDeg:0}°");
    Console.WriteLine("--- settings.json ---");
    Console.WriteLine(File.ReadAllText(UserSettings.FilePath));
    return 0;
}

static void WriteGray(string path, float[] values, int w, int h, bool normalize = true)
{
    float min = 0f, max = 1f;
    if (normalize)
    {
        min = values.Min(); max = values.Max();
        if (max - min < 1e-9f) max = min + 1e-9f;
    }
    var rgb = new byte[w * h * 3];
    for (int i = 0; i < values.Length; i++)
    {
        byte v = (byte)Math.Clamp((values[i] - min) / (max - min) * 255f, 0f, 255f);
        rgb[i * 3] = v; rgb[i * 3 + 1] = v; rgb[i * 3 + 2] = v;
    }
    WriteRgb(path, rgb, w, h);
}

static void WriteRgb(string path, byte[] rgb, int w, int h)
{
    // PPM (P6) then convert via ffmpeg to PNG — avoids image libs
    var ppm = path + ".ppm";
    using (var fs = File.Create(ppm))
    {
        var header = System.Text.Encoding.ASCII.GetBytes($"P6\n{w} {h}\n255\n");
        fs.Write(header);
        fs.Write(rgb);
    }
    var psi = new System.Diagnostics.ProcessStartInfo
    {
        FileName = "ffmpeg",
        Arguments = $"-y -v error -i \"{ppm}\" \"{path}\"",
        UseShellExecute = false,
        CreateNoWindow = true,
    };
    using var p = System.Diagnostics.Process.Start(psi)!;
    p.WaitForExit();
    File.Delete(ppm);
}

static void DumpDepthBinAsPng(string binPath, string pngPath, int cropX = 0, int cropY = 0, int cropW = -1, int cropH = -1, int scale = 1)
{
    var dims = File.ReadAllText(Path.ChangeExtension(binPath, ".txt")).Trim().Split(' ');
    int W = int.Parse(dims[0]), H = int.Parse(dims[1]);
    if (cropW < 0) cropW = W;
    if (cropH < 0) cropH = H;
    var raw = File.ReadAllBytes(binPath);
    var vals = new float[cropW * cropH];
    for (int y = 0; y < cropH; y++)
    {
        int src = ((y + cropY) * W + cropX) * 4;
        int dst = y * cropW;
        for (int x = 0; x < cropW; x++)
        {
            vals[dst + x] = BitConverter.ToSingle(raw, src + x * 4);
        }
    }
    var rgb = new byte[cropW * cropH * 3];
    float mn = vals.Min(), mx = vals.Max();
    if (mx - mn < 1e-9f) mx = mn + 1e-9f;
    for (int i = 0; i < vals.Length; i++)
    {
        byte v = (byte)Math.Clamp((vals[i] - mn) / (mx - mn) * 255f, 0f, 255f);
        rgb[i * 3] = v; rgb[i * 3 + 1] = v; rgb[i * 3 + 2] = v;
    }
    int oW = cropW * scale, oH = cropH * scale;
    var big = new byte[oW * oH * 3];
    for (int y = 0; y < oH; y++)
        for (int x = 0; x < oW; x++)
        {
            int s = ((y / scale) * cropW + (x / scale)) * 3;
            int d = (y * oW + x) * 3;
            big[d] = rgb[s]; big[d + 1] = rgb[s + 1]; big[d + 2] = rgb[s + 2];
        }
    var ppm = pngPath + ".ppm";
    using (var fs = File.Create(ppm))
    {
        var header = System.Text.Encoding.ASCII.GetBytes($"P6\n{oW} {oH}\n255\n");
        fs.Write(header);
        fs.Write(big);
    }
    var psi = new System.Diagnostics.ProcessStartInfo { FileName = "ffmpeg", Arguments = $"-y -v error -i \"{ppm}\" \"{pngPath}\"", UseShellExecute = false, CreateNoWindow = true };
    using var p = System.Diagnostics.Process.Start(psi)!;
    p.WaitForExit();
    File.Delete(ppm);
}
