<#
.SYNOPSIS
    Builds VRify in Release and publishes a self-contained, ready-to-distribute folder.

.DESCRIPTION
    Output: G:\Dev\VRify\publish\VRify\   (the app)
            G:\Dev\VRify\publish\models.zip  (ONNX models, downloaded by the installer)
    - Self-contained: bundles the .NET 9 runtime, no .NET install required on the target.
    - win-x64: matches 99% of desktop Windows machines today.
    - R2R (ReadyToRun): cuts cold-start time from ~3 s to ~0.5 s.
    - Not single-file: keeps native files in their natural layout so AV software and
      the JIT don't get suspicious about a 150-MB "unknown publisher" .exe.

    Models (model.onnx, model_fp16.onnx) are NOT bundled into the app; the installer
    downloads them at install time from the same GitHub release. If models are present
    under .\models\, this script also writes publish\models.zip so the installer can
    use a local file (faster, no network needed for the build).

.EXAMPLE
    .\publish.ps1
#>

$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$out  = Join-Path $root 'publish\VRify'
$zip  = Join-Path $root 'publish\models.zip'
$models = Join-Path $root 'models'

Write-Host "==> Cleaning $out" -ForegroundColor Cyan
if (Test-Path $out) { Remove-Item $out -Recurse -Force }

Write-Host "==> Publishing VRify.App (self-contained, win-x64, R2R)" -ForegroundColor Cyan
& dotnet publish "$root\VRify.App\VRify.App.csproj" `
    --configuration Release `
    --runtime win-x64 `
    --self-contained true `
    --output "$out" `
    /p:PublishReadyToRun=true `
    /p:PublishSingleFile=false `
    /p:DebugType=embedded `
    /p:IncludeNativeLibrariesForSelfExtract=true `
    --nologo

if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed (exit $LASTEXITCODE)" }

# Build models.zip if models are present.
if ((Test-Path "$models\model.onnx") -and (Test-Path "$models\model_fp16.onnx")) {
    Write-Host "==> Building $zip" -ForegroundColor Cyan
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $models, $zip, [System.IO.Compression.CompressionLevel]::Optimal, $false)
    $z = [math]::Round((Get-Item $zip).Length / 1MB, 1)
    Write-Host "    models.zip: $z MB" -ForegroundColor Cyan
} else {
    Write-Host "==> Skipping models.zip (model.onnx not found under .\models\)" -ForegroundColor Yellow
    Write-Host "    Download manually from the v* release and place under .\models\, then re-run." -ForegroundColor Yellow
}

# Size summary.
$size = (Get-ChildItem -Recurse -File $out | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ""
Write-Host "==> Published to $out" -ForegroundColor Green
Write-Host "==> Size: $([math]::Round($size, 1)) MB" -ForegroundColor Green
Write-Host "==> Run: $out\VRify.App.exe" -ForegroundColor Green
