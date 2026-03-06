# setup_data.ps1
# Download datasets for VAD-KD project (Windows PowerShell)
# Usage:
#   .\setup_data.ps1          # LibriParty + MUSAN only (~11 GB)
#   .\setup_data.ps1 -Full    # Include CommonLanguage (~18 GB extra)

param(
    [switch]$Full
)

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VAD-KD Project - Dataset Setup (Windows)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Create data directory
if (-not (Test-Path "data")) { New-Item -ItemType Directory -Path "data" | Out-Null }

# --- LibriParty ---
if (-not (Test-Path "data\LibriParty")) {
    Write-Host "`n[1/3] Downloading LibriParty (~1.4 GB)..." -ForegroundColor Yellow
    $ProgressPreference = 'SilentlyContinue'  # speeds up Invoke-WebRequest
    Invoke-WebRequest -Uri "https://www.dropbox.com/s/ns63xdwmo1agj3r/LibriParty.tar.gz?dl=1" `
        -OutFile "data\LibriParty.tar.gz"
    Write-Host "Extracting LibriParty..."
    tar -xzf "data\LibriParty.tar.gz" -C "data\"
    Remove-Item "data\LibriParty.tar.gz"
    Write-Host "LibriParty ready." -ForegroundColor Green
} else {
    Write-Host "`n[1/3] LibriParty already exists, skipping." -ForegroundColor Gray
}

# --- MUSAN ---
if (-not (Test-Path "data\musan")) {
    Write-Host "`n[2/3] Downloading MUSAN (~10 GB)..." -ForegroundColor Yellow
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri "https://www.openslr.org/resources/17/musan.tar.gz" `
        -OutFile "data\musan.tar.gz"
    Write-Host "Extracting MUSAN..."
    tar -xzf "data\musan.tar.gz" -C "data\"
    Remove-Item "data\musan.tar.gz"
    Write-Host "MUSAN ready." -ForegroundColor Green
} else {
    Write-Host "`n[2/3] MUSAN already exists, skipping." -ForegroundColor Gray
}

# --- CommonLanguage (optional) ---
if ($Full) {
    if (-not (Test-Path "data\common_voice_kpd")) {
        Write-Host "`n[3/3] Downloading CommonLanguage (~6 GB)..." -ForegroundColor Yellow
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri "https://zenodo.org/record/5036977/files/CommonLanguage.tar.gz?download=1" `
            -OutFile "data\CommonLanguage.tar.gz"
        Write-Host "Extracting CommonLanguage..."
        tar -xzf "data\CommonLanguage.tar.gz" -C "data\"
        Remove-Item "data\CommonLanguage.tar.gz"
        Write-Host "CommonLanguage ready." -ForegroundColor Green
    } else {
        Write-Host "`n[3/3] CommonLanguage already exists, skipping." -ForegroundColor Gray
    }
} else {
    Write-Host "`n[3/3] Skipping CommonLanguage (use -Full to download, ~6 GB)" -ForegroundColor Gray
}

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  Datasets ready in .\data\" -ForegroundColor Cyan
Write-Host "    - data\LibriParty\dataset\  (~1.4 GB)" -ForegroundColor White
Write-Host "    - data\musan\               (~10 GB)" -ForegroundColor White
if (Test-Path "data\common_voice_kpd") {
    Write-Host "    - data\common_voice_kpd\    (optional)" -ForegroundColor White
}
Write-Host "============================================" -ForegroundColor Cyan
