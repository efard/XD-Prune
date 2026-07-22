[CmdletBinding()]
param()

# Stop immediately when a command fails. This prevents the script from
# continuing with a partially configured or inconsistent toolchain.
$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
$VenvDirectory = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDirectory "Scripts\python.exe"

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)

    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-CommandAvailable {
    param([Parameter(Mandatory = $true)][string]$CommandName)

    # Get-Command checks whether Windows can resolve the executable from PATH.
    return $null -ne (Get-Command $CommandName -ErrorAction SilentlyContinue)
}

function Refresh-ProcessPath {
    # Installers commonly update the persistent PATH but not the PATH of the
    # currently running PowerShell process. Reloading both scopes lets later
    # checks see newly installed programs without opening another terminal.
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machinePath;$userPath"
}

function Invoke-WingetInstall {
    param(
        [Parameter(Mandatory = $true)][string]$PackageId,
        [Parameter(Mandatory = $true)][string]$DisplayName,
        [string]$OverrideArguments = "",
        [switch]$Force
    )

    Write-Step "Installing $DisplayName"

    $arguments = @(
        "install",
        "--id", $PackageId,
        "--exact",
        "--source", "winget",
        "--accept-package-agreements",
        "--accept-source-agreements"
    )

    if ($OverrideArguments) {
        $arguments += @("--override", $OverrideArguments)
    }

    # --force is used only when Build Tools already exists without the required
    # C++ workload; it asks WinGet to run the installer again so the workload
    # can be added instead of reporting that the package is already installed.
    if ($Force) {
        $arguments += "--force"
    }

    & winget @arguments

    if ($LASTEXITCODE -ne 0) {
        throw "$DisplayName installation failed with WinGet exit code $LASTEXITCODE."
    }

    Refresh-ProcessPath
}

function Get-Python311Executable {
    # Prefer the Python Launcher because it reliably selects Python 3.11 even
    # when another Python version is also installed.
    if (Test-CommandAvailable "py") {
        & py -3.11 --version *> $null
        if ($LASTEXITCODE -eq 0) {
            return "py"
        }
    }

    # These paths cover the standard per-user and all-user Python.org installs.
    $knownPaths = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
        (Join-Path $env:ProgramFiles "Python311\python.exe")
    )

    foreach ($path in $knownPaths) {
        if (Test-Path $path) {
            return $path
        }
    }

    return $null
}

function Invoke-Python311 {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    $pythonCommand = Get-Python311Executable
    if (-not $pythonCommand) {
        throw "Python 3.11 could not be resolved after installation."
    }

    if ($pythonCommand -eq "py") {
        & py -3.11 @Arguments
    }
    else {
        & $pythonCommand @Arguments
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.11 command failed with exit code $LASTEXITCODE."
    }
}

function Test-MsvcBuildTools {
    # cl.exe is normally exposed only after loading a Visual Studio developer
    # shell. vswhere checks the installed workload directly instead of relying
    # on the current PowerShell PATH.
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        return $false
    }

    $installationPath = & $vswhere `
        -latest `
        -products "*" `
        -requires "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" `
        -property installationPath

    return -not [string]::IsNullOrWhiteSpace($installationPath)
}

# Windows capabilities and Visual Studio Build Tools installation require an
# elevated PowerShell session. Failing early avoids a half-completed setup.
$currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
$isAdministrator = $currentPrincipal.IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

if (-not $isAdministrator) {
    throw "Open PowerShell as Administrator, cd to the project folder, and run this script again."
}

if (-not (Test-CommandAvailable "winget")) {
    throw "WinGet is unavailable. Install or update Microsoft App Installer, then run this script again."
}

Write-Step "Project folder"
Write-Host $ProjectRoot

Write-Step "Checking Python 3.11"
if (-not (Get-Python311Executable)) {
    Invoke-WingetInstall `
        -PackageId "Python.Python.3.11" `
        -DisplayName "Python 3.11"
}
else {
    Write-Host "Python 3.11 is already installed." -ForegroundColor Green
}

Write-Step "Checking Git"
if (-not (Test-CommandAvailable "git")) {
    Invoke-WingetInstall `
        -PackageId "Git.Git" `
        -DisplayName "Git"
}
else {
    Write-Host (& git --version) -ForegroundColor Green
}

Write-Step "Checking CMake"
if (-not (Test-CommandAvailable "cmake")) {
    Invoke-WingetInstall `
        -PackageId "Kitware.CMake" `
        -DisplayName "CMake"
}
else {
    Write-Host ((& cmake --version | Select-Object -First 1)) -ForegroundColor Green
}

Write-Step "Checking the Microsoft C/C++ compiler"
if (-not (Test-MsvcBuildTools)) {
    # The VCTools workload installs the MSVC compiler, linker, headers, and
    # recommended Windows SDK components needed by native Python extensions.
    Invoke-WingetInstall `
        -PackageId "Microsoft.VisualStudio.2022.BuildTools" `
        -DisplayName "Visual Studio 2022 C++ Build Tools" `
        -OverrideArguments "--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" `
        -Force

    if (-not (Test-MsvcBuildTools)) {
        throw "Visual Studio Build Tools finished installing, but the MSVC C++ workload was not detected."
    }
}
else {
    Write-Host "MSVC C++ Build Tools are installed." -ForegroundColor Green
}

Write-Step "Checking SSH and SCP"
if (-not (Test-CommandAvailable "ssh") -or -not (Test-CommandAvailable "scp")) {
    $openSshClient = Get-WindowsCapability -Online |
        Where-Object { $_.Name -like "OpenSSH.Client*" } |
        Select-Object -First 1

    if (-not $openSshClient) {
        throw "Windows did not return an OpenSSH Client capability record."
    }

    if ($openSshClient.State -ne "Installed") {
        Add-WindowsCapability -Online -Name $openSshClient.Name | Out-Null
        Refresh-ProcessPath
    }

    if (-not (Test-CommandAvailable "ssh") -or -not (Test-CommandAvailable "scp")) {
        throw "OpenSSH Client was installed, but ssh/scp are not visible in the current PATH."
    }
}

# ssh writes its version to stderr, so invoke it through cmd to capture a
# readable one-line result without treating stderr as a PowerShell failure.
Write-Host (cmd /c "ssh -V 2>&1") -ForegroundColor Green
Write-Host "SCP is available at: $((Get-Command scp).Source)" -ForegroundColor Green

Write-Step "Checking the project virtual environment"
if (Test-Path $VenvDirectory) {
    if (-not (Test-Path $VenvPython)) {
        throw "The .venv folder exists but does not contain Scripts\python.exe. It was not overwritten."
    }

    Write-Host "Existing .venv found." -ForegroundColor Green
}
else {
    # A project-local environment keeps Ultralytics and export packages
    # isolated from system Python and from unrelated projects.
    Invoke-Python311 -Arguments @("-m", "venv", $VenvDirectory)
    Write-Host "Created .venv with Python 3.11." -ForegroundColor Green
}

Write-Step "Installing or verifying Ultralytics and export dependencies"
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to update pip build tools inside .venv."
}

# pip treats this command idempotently: already-satisfied packages remain
# installed, while missing Ultralytics export extras are added to .venv.
& $VenvPython -m pip install "ultralytics[export]"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install Ultralytics export dependencies."
}

# pip check validates that installed package requirements are mutually
# consistent. Import checks then confirm the core ONNX export stack can load.
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "pip detected incompatible or missing Python dependencies."
}

& $VenvPython -c @"
import sys
import ultralytics
import torch
import onnx
import onnxruntime

print(f"Python:       {sys.version.split()[0]}")
print(f"Ultralytics:  {ultralytics.__version__}")
print(f"PyTorch:      {torch.__version__}")
print(f"ONNX:         {onnx.__version__}")
print(f"ONNX Runtime: {onnxruntime.__version__}")
print("Python environment check: PASS")
"@
if ($LASTEXITCODE -ne 0) {
    throw "One or more required Python modules could not be imported."
}

Write-Step "Final system-tool versions"
Write-Host (& git --version)
Write-Host ((& cmake --version | Select-Object -First 1))
Write-Host "MSVC C++ Build Tools: installed"
Write-Host (cmd /c "ssh -V 2>&1")
Write-Host "scp: $((Get-Command scp).Source)"

Write-Host ""
Write-Host "SETUP COMPLETE" -ForegroundColor Green
Write-Host "Activate the environment with:"
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
