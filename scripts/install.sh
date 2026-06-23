#!/bin/bash
# FluxRT installation script (uv-based).
# Run from the repository root: bash scripts/install.sh
set -euo pipefail

# ── colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── sanity-check: running from repo root ──────────────────────────────────────
[ -f "pyproject.toml" ] || die "This script must be run from the FluxRT repository root."

# ── prerequisites ─────────────────────────────────────────────────────────────
log "Checking prerequisites..."

command -v git &>/dev/null || die "'git' is not installed. Install it with your system package manager."
command -v uv  &>/dev/null || die "'uv' is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
git lfs version &>/dev/null || die "'git-lfs' is not installed. Install with: sudo apt install git-lfs  (or brew install git-lfs)"

log "All prerequisites found."

# ── virtual environment (uv, Python 3.12) ─────────────────────────────────────
VENV=".venv"
if [ -x "$VENV/bin/python" ]; then
    log "Virtual environment '${VENV}' already exists."
else
    log "Creating virtual environment '${VENV}' (Python 3.12)..."
    uv venv --python 3.12 "$VENV"
fi

# Target this venv for all subsequent python / uv pip calls without sourcing the
# activate script (which can trip 'set -u').
export VIRTUAL_ENV="$PWD/$VENV"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# ── PyTorch (CUDA 12.8 / Blackwell sm_120 for RTX 50-series) ──────────────────
# Check the build actually carries sm_120 kernels, not just that torch imports.
# A cu128 wheel lists sm_120 regardless of the local GPU and also covers the
# RTX 4090 (sm_89); a CPU or pre-cu128 torch fails on a 5090 ("sm_120 not
# supported"). --upgrade replaces such a build in place.
if python -c "import torch, sys; sys.exit(0 if 'sm_120' in torch.cuda.get_arch_list() else 1)" 2>/dev/null; then
    log "PyTorch with CUDA 12.8 (Blackwell sm_120) already installed."
else
    log "Installing PyTorch with CUDA 12.8 / Blackwell sm_120 support..."
    uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

# ── Python requirements ───────────────────────────────────────────────────────
# Use 'diffusers' as a proxy — it's the heaviest transitive dependency.
if python -c "import diffusers" 2>/dev/null; then
    log "Python requirements already installed."
else
    log "Installing Python requirements from requirements.txt..."
    uv pip install -r requirements.txt
fi

# ── fluxrt package ────────────────────────────────────────────────────────────
if python -c "import fluxrt" 2>/dev/null; then
    log "fluxrt package already installed."
else
    log "Installing fluxrt package in editable mode..."
    uv pip install -e .
fi

# ── model downloads ───────────────────────────────────────────────────────────
# Register LFS hooks for the current user (idempotent).
git lfs install

# clone_or_resume <dir> <url> <sentinel-file> <label>
#   sentinel-file — a large LFS asset that only exists after a complete download.
#   If the directory is present but the sentinel is missing we assume the clone
#   was interrupted and attempt to resume via `git lfs pull`.
clone_or_resume() {
    local dir="$1"
    local url="$2"
    local sentinel="$3"
    local label="$4"

    if [ -f "$sentinel" ]; then
        log "${label}: already downloaded."
        return
    fi

    if [ -d "${dir}/.git" ]; then
        warn "${label}: directory exists but looks incomplete — resuming LFS download..."
        git -C "$dir" pull --ff-only
        git -C "$dir" lfs pull
    elif [ -d "$dir" ]; then
        warn "${label}: directory '${dir}' exists but is not a git repository." \
             "Remove it and re-run to download the model."
        return
    else
        log "Downloading ${label}..."
        git clone "$url" "$dir"
    fi
}

clone_or_resume \
    "RIFE-safetensors" \
    "https://huggingface.co/TensorForger/RIFE-safetensors" \
    "RIFE-safetensors/flownet.safetensors" \
    "RIFE frame-interpolation model"

clone_or_resume \
    "FLUX.2-klein-4B" \
    "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B" \
    "FLUX.2-klein-4B/transformer/diffusion_pytorch_model.safetensors" \
    "FLUX.2-klein-4B base model"

clone_or_resume \
    "FLUX.2-klein-4B-int8" \
    "https://huggingface.co/aydin99/FLUX.2-klein-4B-int8" \
    "FLUX.2-klein-4B-int8/diffusion_pytorch_model.safetensors" \
    "FLUX.2-klein-4B int8 model"

# ── extension models (enabled per-config; small, downloaded by default) ───────
clone_or_resume \
    "taef2" \
    "https://huggingface.co/madebyollin/taef2" \
    "taef2/taef2.safetensors" \
    "TAEF2 tiny-VAE model (extension)"

clone_or_resume \
    "FlowUpscaler" \
    "https://huggingface.co/TensorForger/FlowUpscaler" \
    "FlowUpscaler/flow_upscaler.safetensors" \
    "Flow Upscaler model (extension)"

# ── verify GPU detection ──────────────────────────────────────────────────────
log "Detected GPU(s):"
python -c "from fluxrt.utils.scan_hardware import scan_hardware; import json; print(json.dumps(scan_hardware()['gpu'], indent=2))" \
    || warn "Could not query the GPU — check the PyTorch install above."

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
log "${BOLD}Installation complete.${NC}"
warn "The GUI requires v4l2loopback (kernel module) for virtual-webcam output."
warn "      Install it with your package manager, then: sudo modprobe v4l2loopback"
log "Activate the environment and start:  ${BOLD}source ${VENV}/bin/activate${NC}"
log "Then run, for example:               ${BOLD}python scripts/run_gradio_demo.py${NC}"
