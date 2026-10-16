# VRify — 2D → VR180/3D SBS converter

Turns regular 2D videos into VR180 equirectangular stereo (SBS) or flat side-by-side
stereo pairs, ready for any VR headset (Meta Quest, PICO, Valve Index, Phone VR…).

A neural net (Depth-Anything ONNX) estimates per-frame depth, then a fast GPU pipeline
forward-warps the right eye, fills disocclusion holes, and projects into VR180
equirectangular (or packs as flat SBS).

## Features

- **VR180 equirectangular SBS** (4320×2160, 180° lon × 180° lat per eye, 2:1 aspect)
- **Flat side-by-side SBS** (1920×1080 × 2 = 3840×1080)
- **GPU pipeline** (DirectML / Direct3D 11) on any modern NVIDIA / AMD / Intel GPU
- **CPU fallback** if no compatible GPU is available
- **Adjustable stereo strength** (0.5 – 6.0 %) and **virtual-screen FOV** (70° – 175°)
- **HEVC (H.265) / H.264** output, MP4 / MKV
- **Drag & drop**, last-used file paths, remembered window placement
- **Light/dark UI**, presets, progress with ETA

## Installation

Download the latest installer from the [Releases](../../releases) page.

* **Windows 10/11 x64**
* No .NET install required (self-contained build)
* ~300 MB on disk, ~180 MB installer

The installer will offer to download the ONNX depth models (~150 MB) from
the same GitHub release. You need an internet connection on the target
machine for this step. If the download fails the app will still install
and start, but conversion will need `model.onnx` next to the exe
(see *Manual install* below).

### Manual install (no internet at install time)

1. Download **both** `VRify-x.y.z-setup.exe` and `models.zip` from Releases
2. Run the installer
3. Unzip `models.zip` into `C:\Program Files\VRify\models\`
   (the installer creates the `models\` folder even if the download failed)

## Quick start

1. Launch **VRify**
2. Drag a 2D video into the window (or click *Browse…*)
3. Pick a preset: **Soft** (1.5%), **Standard** (2.5%), **Strong** (3.5%)
4. Click **Preview** to see the first frame in stereo
5. Click **Convert** to render the full video

Output file lands next to the source with `_VR180_SBS.mp4` (or `_flatSBS.mp4`) suffix.

In your VR player (Skybox, Pigasus, DeoVR, …), pick **180 SBS / Equirectangular** layout.

## CLI

```powershell
VRify.Cli <input.mp4> [output.mp4] [strength%] [smoothing 0-0.9] [flat|180] [fov]
VRify.Cli debug  <input> <frameA> <frameB> <strength%> <outdir>   # dump depth/hole maps
VRify.Cli ramptest <dir> [y0 y1] [x0 x1]                          # measure depth ramp width
VRify.Cli settings [--set fov=110 strength=2.5 smoothing=0 hevc=true mode=180] [--reset]
```

The GUI and the CLI share the same settings file at `%APPDATA%\VRify\settings.json`.

## Build from source

Requires .NET 9 SDK and a Windows machine (DirectX 11).

```powershell
# 1. Drop a depth model into .\models\ (download separately or use the bundled one):
#    model.onnx          (~99 MB) — fp32
#    model_fp16.onnx     (~50 MB) — fp16, faster on GPU

# 2. Build
dotnet build VRify.sln -c Release

# 3. Run (needs models\model.onnx in the same folder as the .exe)
.\VRify.App\bin\Release\net9.0-windows\VRify.App.exe

# 4. Produce a self-contained build for distribution
.\publish.ps1
#  -> publish\VRify\         (~300 MB, the app)
#  -> publish\models.zip     (~130 MB, models for the installer to download)
#
# Then run InnoSetup to make the installer:
#   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
#  -> publish\VRify-0.1.0-setup.exe   (~180 MB)
```

## License

[MIT](LICENSE)
