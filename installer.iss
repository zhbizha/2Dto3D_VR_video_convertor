; VRify — InnoSetup 6 installer script.
; Builds a Windows installer that puts the self-contained app into
; %ProgramFiles%\VRify and creates desktop / start-menu shortcuts.
; At install time, the script downloads the ONNX depth models from the same
; GitHub release as models.zip and unpacks them under {app}\models\.
;
; Prerequisite: run .\publish.ps1 first so that .\publish\VRify\ exists
; and (optionally) .\publish\models.zip if you want the installer to fall
; back to a local file.

#define MyAppName "VRify"
#define MyAppPublisher "Maksim Kuchinskii"
#define MyAppURL "https://github.com/zhbizha/2Dto3D_VR_video_convertor"
#define MyAppExeName "VRify.App.exe"
#define MyAppVersion "0.1.0"
#define ModelsUrl "https://github.com/zhbizha/2Dto3D_VR_video_convertor/releases/download/v{#MyAppVersion}/models.zip"

[Setup]
; Unique per build. Append a random suffix if you sign installers.
AppId={{B5A8F2C3-4D7E-4B9A-9F1C-2E6D8A1B3C5D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE
InfoBeforeFile=
OutputDir=publish
OutputBaseFilename=VRify-{#MyAppVersion}-setup
SetupIconFile=
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
VersionInfoVersion={#MyAppVersion}
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1

[Files]
; Everything publish.ps1 produced. The Source path is the publish output folder;
; DestDir is the install location, so all files end up under %ProgramFiles%\VRify.
Source: "publish\VRify\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Wipe per-user settings + cached models on uninstall so a reinstall starts clean.
Type: filesandordirs; Name: "{userappdata}\VRify"
Type: filesandordirs; Name: "{app}\models"

[Code]
{ Download the ONNX models zip from the GitHub release and extract it into
  {app}\models\. If models are already present (re-install or manual copy)
  the download is skipped. Runs *after* files are installed so the destination
  directory exists with write access. }
const
  ModelsUrl = '{#ModelsUrl}';

function ShouldFetchModels(): Boolean;
begin
  Result := not FileExists(ExpandConstant('{app}\models\model.onnx'));
end;

procedure FetchAndExtractModels();
var
  ZipPath: String;
  ModelsDir: String;
  Cmd: String;
  ExitCode: Integer;
begin
  ModelsDir := ExpandConstant('{app}\models');
  CreateDir(ModelsDir);

  if not ShouldFetchModels() then
  begin
    Log('Models already present, skipping download.');
    exit;
  end;

  ZipPath := ExpandConstant('{tmp}\vrify-models.zip');
  DeleteFile(ZipPath);

  Log(Format('Downloading models from %s to %s', [ModelsUrl, ZipPath]));
  if not URLDownloadToFile(nil, ModelsUrl, ZipPath, 0, nil) then
  begin
    MsgBox('Не удалось скачать модели нейронной сети из GitHub. ' +
           'Приложение запустится, но конвертация потребует файл model.onnx. ' +
           'Скачайте models.zip из раздела Releases и распакуйте в папку установки\models.',
           mbError, MB_OK);
    exit;
  end;

  // Unzip via PowerShell (built into Windows 10/11). 7-Zip would be lighter
  // but adds a dependency. We use Expand-Archive which is reliable enough
  // for our ~150 MB zip.
  Cmd := Format(
    'powershell -NoProfile -ExecutionPolicy Bypass -Command ' +
    '"Expand-Archive -LiteralPath ''%s'' -DestinationPath ''%s'' -Force"',
    [ZipPath, ModelsDir]);
  if not Exec('powershell.exe', Cmd, '', SW_HIDE, ewWaitUntilTerminated, ExitCode) then
  begin
    MsgBox('Не удалось распаковать архив моделей. Скачайте models.zip из ' +
           'Releases вручную и распакуйте в ' + ModelsDir + '.',
           mbError, MB_OK);
    exit;
  end;

  DeleteFile(ZipPath);
  Log('Models extracted successfully.');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    FetchAndExtractModels();
end;
