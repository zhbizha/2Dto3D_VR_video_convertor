using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using VRify.Core;

namespace VRify.App;

public partial class MainWindow : Window
{
    private VrConverter? _converter;
    private CancellationTokenSource? _cts;
    private bool _busy;
    private DateTime _convertStart;
    private UserSettings _settings = new();

    public MainWindow()
    {
        InitializeComponent();
        Loaded += (_, _) => ApplySettings(UserSettings.Load());
    }

    /// <summary>
    /// Returns the cached <see cref="VrConverter"/>, loading the model on first use.
    /// Must be called from the UI thread the first time, because it writes to
    /// <see cref="StatusText"/>. Subsequent calls are thread-safe (the cached reference
    /// is read-only).
    /// </summary>
    private VrConverter GetConverter()
    {
        if (_converter != null) return _converter;
        StatusText.Text = "Загрузка модели глубины…";
        var model = VrConverter.FindModel();
        _converter = new VrConverter(model);
        StatusText.Text = $"Модель загружена ({_converter.Provider})";
        return _converter;
    }

    private void BrowseInput_Click(object sender, RoutedEventArgs e) => PickInputFile();

    private void BrowseOutput_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new SaveFileDialog
        {
            Title = "Куда сохранить SBS 180",
            Filter = "MP4|*.mp4|Matroska|*.mkv",
            FileName = string.IsNullOrWhiteSpace(OutputBox.Text) ? null : Path.GetFileName(OutputBox.Text),
            InitialDirectory = string.IsNullOrWhiteSpace(OutputBox.Text) ? null : Path.GetDirectoryName(OutputBox.Text),
        };
        if (dlg.ShowDialog() == true) OutputBox.Text = dlg.FileName;
    }

    private void Window_DragOver(object sender, DragEventArgs e)
    {
        e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop) ? DragDropEffects.Copy : DragDropEffects.None;
        e.Handled = true;
    }

    private void Window_Drop(object sender, DragEventArgs e)
    {
        if (!e.Data.GetDataPresent(DataFormats.FileDrop)) return;
        var files = (string[])e.Data.GetData(DataFormats.FileDrop);
        if (files.Length == 0) return;
        // Take the first file (UI is single-file at a time).
        var path = files[0];
        if (File.Exists(path)) SetInputFile(path);
    }

    private void PickInputFile()
    {
        var dlg = new OpenFileDialog
        {
            Title = "Выберите 2D видео",
            Filter = "Видео|*.mp4;*.mkv;*.mov;*.avi;*.webm;*.m4v|Все файлы|*.*",
            InitialDirectory = !string.IsNullOrWhiteSpace(_settings.LastInputPath) && File.Exists(_settings.LastInputPath)
                ? Path.GetDirectoryName(_settings.LastInputPath)
                : null,
        };
        if (dlg.ShowDialog() != true) return;
        SetInputFile(dlg.FileName);
    }

    private void SetInputFile(string path)
    {
        InputBox.Text = path;
        // Suggest an output next to the source, but respect the last-used output folder if any.
        var name = Path.GetFileNameWithoutExtension(path);
        var dir = !string.IsNullOrWhiteSpace(_settings.LastOutputPath) && Directory.Exists(Path.GetDirectoryName(_settings.LastOutputPath))
            ? Path.GetDirectoryName(_settings.LastOutputPath)!
            : (Path.GetDirectoryName(path) ?? "");
        var suffix = ProjectionCombo.SelectedIndex == 1 ? "_VR180_SBS" : "_flatSBS";
        OutputBox.Text = Path.Combine(dir, $"{name}{suffix}.mp4");
        StatusText.Text = $"Загружено: {Path.GetFileName(path)}";
        UpdateButtons();
    }

    private void Preset_Click(object sender, RoutedEventArgs e)
    {
        if (sender is not System.Windows.Controls.Button b || b.Tag is not string tag) return;
        if (!float.TryParse(tag, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var v)) return;
        StrengthSlider.Value = v;
        SmoothingSlider.Value = 0;     // presets always use zero smoothing (best stereo)
        FovSlider.Value = 110;         // and the user's preferred FOV
    }

    private void OpenFolder_Click(object sender, RoutedEventArgs e)
    {
        var path = !string.IsNullOrWhiteSpace(OutputBox.Text) && File.Exists(OutputBox.Text)
            ? OutputBox.Text
            : (!string.IsNullOrWhiteSpace(InputBox.Text) && File.Exists(InputBox.Text) ? InputBox.Text : null);
        if (path == null)
        {
            StatusText.Text = "Нет файла для открытия";
            return;
        }
        try
        {
            Process.Start("explorer.exe", $"/select,\"{path}\"");
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Не удалось открыть проводник",
                MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    private void Setting_Changed(object sender, RoutedPropertyChangedEventArgs<double> e) => RefreshLabels();

    /// <summary>Pushes the persisted values into the controls (clamped to each range).</summary>
    private void ApplySettings(UserSettings s)
    {
        _settings = s;
        s.Normalize();
        StrengthSlider.Value = s.DepthStrengthPercent;
        SmoothingSlider.Value = s.TemporalSmoothing;
        HevcCheck.IsChecked = s.Hevc;
        ProjectionCombo.SelectedIndex = s.Mode == StereoMode.Vr180 ? 1 : 0;
        FovSlider.Value = s.ScreenFovDeg;
        // Restore file paths and window placement.
        if (!string.IsNullOrWhiteSpace(s.LastInputPath) && File.Exists(s.LastInputPath))
            InputBox.Text = s.LastInputPath;
        if (!string.IsNullOrWhiteSpace(s.LastOutputPath))
            OutputBox.Text = s.LastOutputPath;
        if (s.WindowWidth >= 400) Width = s.WindowWidth;
        if (s.WindowHeight >= 300) Height = s.WindowHeight;
        if (s.WindowLeft.HasValue && s.WindowTop.HasValue
            && s.WindowLeft.Value >= 0 && s.WindowTop.Value >= 0
            && s.WindowLeft.Value < SystemParameters.VirtualScreenWidth
            && s.WindowTop.Value < SystemParameters.VirtualScreenHeight)
        {
            WindowStartupLocation = WindowStartupLocation.Manual;
            Left = s.WindowLeft.Value;
            Top = s.WindowTop.Value;
        }
        if (s.WindowMaximized) WindowState = WindowState.Maximized;
        RefreshLabels();
        Projection_Changed(this, null!);
        UpdateButtons();
    }

    /// <summary>Reads the current control values into a settings object (file paths + window placement included).</summary>
    private UserSettings CollectSettings()
    {
        var s = new UserSettings
        {
            DepthStrengthPercent = (float)StrengthSlider.Value,
            TemporalSmoothing = (float)SmoothingSlider.Value,
            Hevc = HevcCheck.IsChecked == true,
            Mode = ProjectionCombo.SelectedIndex == 1 ? StereoMode.Vr180 : StereoMode.FlatSbs,
            ScreenFovDeg = FovSlider.Value,
            LastInputPath = string.IsNullOrWhiteSpace(InputBox.Text) ? null : InputBox.Text,
            LastOutputPath = string.IsNullOrWhiteSpace(OutputBox.Text) ? null : OutputBox.Text,
            WindowWidth = WindowState == WindowState.Maximized ? _settings.WindowWidth : Width,
            WindowHeight = WindowState == WindowState.Maximized ? _settings.WindowHeight : Height,
            WindowMaximized = WindowState == WindowState.Maximized,
        };
        if (WindowState == WindowState.Normal)
        {
            s.WindowLeft = Left;
            s.WindowTop = Top;
        }
        else
        {
            s.WindowLeft = RestoreBounds.Left;
            s.WindowTop = RestoreBounds.Top;
        }
        return s;
    }

    /// <summary>
    /// Refreshes the value labels. Guards against the controls not existing yet: WPF fires
    /// ValueChanged while InitializeComponent is still building the tree (slider precedes label).
    /// </summary>
    private void RefreshLabels()
    {
        if (StrengthLabel != null) StrengthLabel.Text = $"{StrengthSlider.Value:0.0} %";
        if (SmoothingLabel != null) SmoothingLabel.Text = $"{SmoothingSlider.Value:0.0}";
        if (FovLabel != null) FovLabel.Text = $"{FovSlider.Value:0}°";
        UpdatePresetButtons();
    }

    /// <summary>Highlights the preset button whose value matches the current strength.</summary>
    private void UpdatePresetButtons()
    {
        void Highlight(System.Windows.Controls.Button b, float target)
        {
            bool active = Math.Abs((float)StrengthSlider.Value - target) < 0.01f
                && Math.Abs((float)SmoothingSlider.Value) < 0.01f
                && Math.Abs(FovSlider.Value - 110.0) < 0.5f;
            b.Background = new System.Windows.Media.SolidColorBrush(
                active ? (System.Windows.Media.Color)System.Windows.Media.ColorConverter.ConvertFromString("#89B4FA")!
                       : (System.Windows.Media.Color)System.Windows.Media.ColorConverter.ConvertFromString("#45475A")!);
            b.Foreground = new System.Windows.Media.SolidColorBrush(
                active ? (System.Windows.Media.Color)System.Windows.Media.ColorConverter.ConvertFromString("#1E1E2E")!
                       : (System.Windows.Media.Color)System.Windows.Media.ColorConverter.ConvertFromString("#CDD6F4")!);
        }
        if (PresetSoft != null) Highlight(PresetSoft, 1.5f);
        if (PresetNormal != null) Highlight(PresetNormal, 2.5f);
        if (PresetStrong != null) Highlight(PresetStrong, 3.5f);
    }

    private ConversionSettings BuildSettings(string input, string output) => new(
        input,
        output,
        DepthStrengthPercent: (float)StrengthSlider.Value,
        TemporalSmoothing: (float)SmoothingSlider.Value,
        Hevc: HevcCheck.IsChecked == true,
        Mode: ProjectionCombo.SelectedIndex == 1 ? StereoMode.Vr180 : StereoMode.FlatSbs,
        ScreenFovDeg: FovSlider.Value);

    private void Projection_Changed(object sender, SelectionChangedEventArgs e)
    {
        if (FovSlider == null || FovText == null) return;
        bool vr = ProjectionCombo.SelectedIndex == 1;
        FovSlider.IsEnabled = vr;
        FovText.Opacity = vr ? 1.0 : 0.4;
        FovLabel.Opacity = vr ? 1.0 : 0.4;
    }

    private void UpdateButtons()
    {
        bool hasFiles = File.Exists(InputBox.Text) && !string.IsNullOrWhiteSpace(OutputBox.Text);
        PreviewButton.IsEnabled = hasFiles && !_busy;
        ConvertButton.IsEnabled = hasFiles && !_busy;
        OpenFolderButton.IsEnabled = (File.Exists(InputBox.Text) || File.Exists(OutputBox.Text)) && !_busy;
        CancelButton.IsEnabled = _busy;
    }

    private async void Preview_Click(object sender, RoutedEventArgs e)
    {
        if (_busy) return;
        SetBusy(true, "Генерация превью (первый кадр)…");
        try
        {
            var input = InputBox.Text;
            var settings = BuildSettings(input, "");
            CollectSettings().Save();   // remember the preset for the next launch
            // Touch the converter in the UI thread so the first-time model load happens here.
            var conv = GetConverter();
            var (sbs, pw, ph) = await Task.Run(() =>
            {
                var info = VideoInfo.Probe(input);
                // Decode just the first frame
                var frame = new byte[info.Width * info.Height * 3];
                using var reader = new FrameReader(input, info.Width, info.Height);
                if (reader.ReadFrame(frame) != frame.Length)
                    throw new InvalidOperationException("Не удалось прочитать первый кадр");
                var r = conv.ProcessPreviewFrame(frame, info.Width, info.Height, settings);
                return (r.Sbs, r.Width, r.Height);
            });

            PreviewImage.Source = ToBitmapSource(sbs, pw, ph);
            StatusText.Text = settings.Mode == StereoMode.Vr180
                ? $"Превью VR180 готов — {pw}×{ph}"
                : $"Превью готово — {pw}×{ph} (левый = оригинал, правый = синтез)";
            InfoText.Visibility = Visibility.Collapsed;
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Ошибка превью", MessageBoxButton.OK, MessageBoxImage.Error);
            StatusText.Text = "Ошибка";
        }
        finally { SetBusy(false); }
    }

    private async void Convert_Click(object sender, RoutedEventArgs e)
    {
        if (_busy) return;
        var settings = BuildSettings(InputBox.Text, OutputBox.Text);
        CollectSettings().Save();   // persist before the long run so a crash/cancel keeps it

        // Touch the converter in the UI thread so the first-time model load (which
        // writes to StatusText) happens here, not from the worker thread.
        var converter = GetConverter();

        _cts = new CancellationTokenSource();
        _convertStart = DateTime.UtcNow;
        SetBusy(true, "Конвертация…");
        var progress = new Progress<ConversionProgress>(p =>
        {
            ProgressBar.Value = p.Percent;
            var elapsed = (DateTime.UtcNow - _convertStart).TotalSeconds;
            var eta = p.Fps > 0.1 ? TimeSpan.FromSeconds((p.FramesTotal - p.FramesDone) / p.Fps) : TimeSpan.Zero;
            StatusText.Text =
                $"Кадр {p.FramesDone}/{p.FramesTotal} • {p.Percent:0.0}% • {p.Fps:0.0} fps" +
                (elapsed > 2 ? $" • прошло {Format(elapsed)} • осталось {Format(eta.TotalSeconds)}" : "");
        });

        try
        {
            await Task.Run(() =>
            {
                converter.Convert(settings, progress, _cts.Token);
            }, _cts.Token);

            ProgressBar.Value = 100;
            var dur = (DateTime.UtcNow - _convertStart).TotalSeconds;
            StatusText.Text = $"Готово за {Format(dur)}: {Path.GetFileName(OutputBox.Text)}";
            UpdateButtons();
        }
        catch (OperationCanceledException)
        {
            StatusText.Text = "Отменено";
            try { File.Delete(OutputBox.Text); } catch { /* partial file */ }
        }
        catch (Exception ex)
        {
            // Dump the full exception chain to a log so we can diagnose cross-thread
            // and other hard-to-reproduce issues without losing the stack.
            try
            {
                var log = Path.Combine(Path.GetTempPath(), "vrify_error.log");
                File.WriteAllText(log,
                    $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {ex}\n\n" +
                    $"--- Input:  {InputBox.Text}\n" +
                    $"--- Output: {OutputBox.Text}\n" +
                    $"--- Settings: strength={StrengthSlider.Value:0.0} smoothing={SmoothingSlider.Value:0.0} " +
                    $"fov={FovSlider.Value:0} mode={(ProjectionCombo.SelectedIndex == 1 ? "VR180" : "FlatSBS")} hevc={HevcCheck.IsChecked}\n");
            }
            catch { /* never block the user dialog */ }
            MessageBox.Show(this, $"{ex.Message}\n\nЛог: {Path.Combine(Path.GetTempPath(), "vrify_error.log")}",
                "Ошибка конвертации", MessageBoxButton.OK, MessageBoxImage.Error);
            StatusText.Text = "Ошибка";
        }
        finally
        {
            _cts?.Dispose();
            _cts = null;
            SetBusy(false);
        }
    }

    private static string Format(double seconds)
    {
        if (seconds < 0 || double.IsNaN(seconds) || double.IsInfinity(seconds)) return "—";
        var t = TimeSpan.FromSeconds(seconds);
        if (t.TotalHours >= 1) return $"{(int)t.TotalHours}ч {t.Minutes}м";
        if (t.TotalMinutes >= 1) return $"{t.Minutes}м {t.Seconds}с";
        return $"{t.Seconds}с";
    }

    private void Cancel_Click(object sender, RoutedEventArgs e) => _cts?.Cancel();

    private void SetBusy(bool busy, string? status = null)
    {
        _busy = busy;
        if (status != null) StatusText.Text = status;
        if (!busy) ProgressBar.Value = 0;
        UpdateButtons();
    }

    private static BitmapSource ToBitmapSource(byte[] rgb, int width, int height)
    {
        var bmp = new WriteableBitmap(width, height, 96, 96, System.Windows.Media.PixelFormats.Rgb24, null);
        bmp.WritePixels(new Int32Rect(0, 0, width, height), rgb, width * 3, 0);
        bmp.Freeze();
        return bmp;
    }

    private void Window_Loaded(object sender, RoutedEventArgs e) { /* ApplySettings runs in ctor via Loaded event */ }

    private void Window_Closing(object? sender, CancelEventArgs e)
    {
        CollectSettings().Save();
        _cts?.Cancel();
        _converter?.Dispose();
    }
}
