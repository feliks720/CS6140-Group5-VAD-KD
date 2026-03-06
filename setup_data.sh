#!/bin/bash
# Download datasets for VAD training
# Run: bash setup_data.sh

set -e
mkdir -p data

echo "============================================"
echo "  VAD-KD Project - Dataset Setup"
echo "============================================"

# --- LibriParty ---
if [ ! -d "data/LibriParty" ]; then
    echo "[1/3] Downloading LibriParty..."
    wget -q --show-progress -O data/LibriParty.tar.gz \
        "https://www.dropbox.com/s/ns63xdwmo1agj3r/LibriParty.tar.gz?dl=1"
    echo "Extracting LibriParty..."
    tar -xzf data/LibriParty.tar.gz -C data/
    rm data/LibriParty.tar.gz
    echo "LibriParty ready."
else
    echo "[1/3] LibriParty already exists, skipping."
fi

# --- MUSAN ---
if [ ! -d "data/musan" ]; then
    echo "[2/3] Downloading MUSAN..."
    wget -q --show-progress -O data/musan.tar.gz \
        "https://www.openslr.org/resources/17/musan.tar.gz"
    echo "Extracting MUSAN..."
    tar -xzf data/musan.tar.gz -C data/
    rm data/musan.tar.gz
    echo "MUSAN ready."
else
    echo "[2/3] MUSAN already exists, skipping."
fi

# --- CommonLanguage (OPTIONAL, ~6GB download, ~18GB extracted) ---
# Only needed for full SpeechBrain training recipe with multilingual augmentation.
# For KD experiments, LibriParty + MUSAN is sufficient.
if [ "$1" = "--full" ]; then
    if [ ! -d "data/common_voice_kpd" ]; then
        echo "[3/3] Downloading CommonLanguage (optional)..."
        wget -q --show-progress -O data/CommonLanguage.tar.gz \
            "https://zenodo.org/record/5036977/files/CommonLanguage.tar.gz?download=1"
        echo "Extracting CommonLanguage..."
        tar -xzf data/CommonLanguage.tar.gz -C data/
        rm data/CommonLanguage.tar.gz
        echo "CommonLanguage ready."
    else
        echo "[3/3] CommonLanguage already exists, skipping."
    fi
else
    echo "[3/3] Skipping CommonLanguage (pass --full to download, ~6GB)."
fi

echo ""
echo "Datasets ready in ./data/"
echo "  - data/LibriParty/dataset/  (~1.4 GB)"
echo "  - data/musan/               (~10 GB)"
if [ -d "data/common_voice_kpd" ]; then
    echo "  - data/common_voice_kpd/    (optional, ~18 GB)"
fi
echo ""
echo "Total disk usage:"
du -sh data/ 2>/dev/null || echo "  (run after download completes)"
