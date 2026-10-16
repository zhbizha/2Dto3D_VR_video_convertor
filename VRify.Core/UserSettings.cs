using System.Text.Json;
using System.Text.Json.Serialization;

namespace VRify.Core;

/// <summary>
/// User preferences for the desktop app, persisted as JSON under
/// <c>%APPDATA%\VRify\settings.json</c>. The CLI exposes a <c>settings</c> subcommand
/// that loads/saves the same file, so both front-ends share one configuration.
///
/// Defaults reproduce the reference preset the user approved for VR180 output:
/// strength 2.5 %, temporal smoothing 0, VR180 equirect, 110° virtual-screen FOV.
/// </summary>
public sealed class UserSettings
{
    public float DepthStrengthPercent { get; set; } = 2.5f;
    public float TemporalSmoothing { get; set; } = 0f;
    public bool Hevc { get; set; } = true;

    [JsonConverter(typeof(JsonStringEnumConverter))]
    public StereoMode Mode { get; set; } = StereoMode.Vr180;

    public double ScreenFovDeg { get; set; } = 110.0;

    // Interface language: "auto" (follow OS), "ru", or "en". Applied at startup.
    public string Language { get; set; } = "auto";

    // Last-used file paths (sticky across launches).
    public string? LastInputPath { get; set; }
    public string? LastOutputPath { get; set; }

    // Window placement (persists size and position so the app opens where the user left it).
    public double WindowWidth { get; set; } = 900;
    public double WindowHeight { get; set; } = 640;
    public double? WindowLeft { get; set; }
    public double? WindowTop { get; set; }
    public bool WindowMaximized { get; set; }

    /// <summary>Directory holding the settings file (%APPDATA%\VRify).</summary>
    [JsonIgnore]
    public static string Directory =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "VRify");

    /// <summary>Full path of the settings file.</summary>
    [JsonIgnore]
    public static string FilePath => Path.Combine(Directory, "settings.json");

    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = true };

    /// <summary>
    /// Loads settings, falling back to defaults when the file is missing, unreadable or
    /// corrupt (a half-written file must never block startup).
    /// </summary>
    public static UserSettings Load()
    {
        try
        {
            if (File.Exists(FilePath))
            {
                var loaded = JsonSerializer.Deserialize<UserSettings>(File.ReadAllText(FilePath), JsonOpts);
                if (loaded != null) { loaded.Normalize(); return loaded; }
            }
        }
        catch { /* missing/corrupt/partial file: use defaults */ }
        return new UserSettings();
    }

    /// <summary>Writes the settings file. Best-effort: never throws into the caller.</summary>
    public void Save()
    {
        try
        {
            System.IO.Directory.CreateDirectory(Directory);
            File.WriteAllText(FilePath, JsonSerializer.Serialize(this, JsonOpts));
        }
        catch { /* settings are a convenience; a failed write must not abort a conversion */ }
    }

    /// <summary>Clamps every value into the range the UI/engine supports.</summary>
    public void Normalize()
    {
        DepthStrengthPercent = Math.Clamp(DepthStrengthPercent, 0.5f, 6f);
        TemporalSmoothing = Math.Clamp(TemporalSmoothing, 0f, 0.9f);
        ScreenFovDeg = Math.Clamp(ScreenFovDeg, 60.0, 175.0);
    }
}
