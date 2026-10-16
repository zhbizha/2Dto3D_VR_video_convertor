using System.Configuration;
using System.Data;
using System.IO;
using System.Windows;
using System.Windows.Threading;

namespace VRify.App;

/// <summary>
/// Interaction logic for App.xaml
/// </summary>
public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        DispatcherUnhandledException += OnUiThreadException;
        AppDomain.CurrentDomain.UnhandledException += OnDomainException;
    }

    private void OnUiThreadException(object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        DumpFatal(e.Exception, "UI thread");
        e.Handled = true;   // keep the app alive long enough to show the user what happened
    }

    private void OnDomainException(object sender, UnhandledExceptionEventArgs e)
    {
        if (e.ExceptionObject is Exception ex) DumpFatal(ex, "AppDomain");
    }

    private static void DumpFatal(Exception ex, string where)
    {
        try
        {
            var log = Path.Combine(Path.GetTempPath(), "vrify_fatal.log");
            File.AppendAllText(log,
                $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {where}: {ex}\n\n");
        }
        catch { }
    }
}

