#ifndef AppVersion
  #error AppVersion must be supplied by installer/build.ps1
#endif
#ifndef StageDir
  #error StageDir must be supplied by installer/build.ps1
#endif

#define ProjectRoot SourcePath + "\.."

[Setup]
AppId={{C64355D5-8A1F-4A10-8DBB-7E72BCE2C297}
AppName=Warlock Studio
AppVersion={#AppVersion}
AppPublisher=Warlock Studio
DefaultDirName={localappdata}\Programs\Warlock Studio
DefaultGroupName=Warlock Studio
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
DiskSpanning=yes
DiskSliceSize=2100000000
OutputDir={#ProjectRoot}\dist
OutputBaseFilename=WarlockSetup-v{#AppVersion}
SetupIconFile={#ProjectRoot}\src\warlock\assets\icon.ico
UninstallDisplayIcon={app}\src\warlock\assets\icon.ico
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[InstallDelete]
Type: filesandordirs; Name: "{app}\python\Lib\site-packages"
Type: filesandordirs; Name: "{app}\src"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Warlock Studio"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m warlock"; WorkingDir: "{app}"
Name: "{group}\Warlock Doctor"; Filename: "{app}\bin\warlock-doctor.cmd"; WorkingDir: "{app}"
Name: "{userdesktop}\Warlock Studio"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m warlock"; WorkingDir: "{app}"; Tasks: desktopicon

[UninstallDelete]
Type: filesandordirs; Name: "{app}\src"
Type: filesandordirs; Name: "{app}\python"

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataPath: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataPath := AddBackslash(GetEnv('USERPROFILE')) + '.warlock';
    MsgBox(
      'Warlock Studio was removed. Your assets and downloaded models remain at ' + DataPath + '.',
      mbInformation,
      MB_OK
    );
  end;
end;
