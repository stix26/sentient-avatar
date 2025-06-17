#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print with color
print_step() {
    echo -e "${GREEN}==>${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

print_error() {
    echo -e "${RED}Error:${NC} $1"
    exit 1
}

# Check disk space (minimum 50GB required)
check_disk_space() {
    local required_space=53687091200  # 50GB in bytes
    local available_space=$(df -B1 . | awk 'NR==2 {print $4}')
    
    if [ "$available_space" -lt "$required_space" ]; then
        print_error "Insufficient disk space. At least 50GB required."
    fi
}

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    # Check if Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        ARCH="arm64"
    else
        ARCH="x86_64"
    fi
elif [[ -f /etc/os-release ]]; then
    . /etc/os-release
    if [[ "$ID" == "ubuntu" && "$VERSION_ID" == "22.04" ]]; then
        OS="ubuntu"
    else
        print_error "Unsupported Linux distribution. Please use Ubuntu 22.04."
    fi
else
    print_error "Unsupported operating system. Please use macOS or Ubuntu 22.04."
fi

# Check disk space before proceeding
check_disk_space

# Create project directory structure
print_step "Creating project directory structure..."
mkdir -p ~/sentient-avatar/{src,storage,models,logs}
cd ~/sentient-avatar

# Install system dependencies
print_step "Installing system dependencies..."

if [[ "$OS" == "macos" ]]; then
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        print_step "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    # Install system packages
    print_step "Installing system packages via Homebrew..."
    brew install \
        git \
        curl \
        ffmpeg \
        cmake \
        node@18 \
        python@3.11 \
        rust \
        pkg-config \
        portaudio \
        wget \
        unzip \
        jq

    # Add Node.js to PATH
    echo 'export PATH="/usr/local/opt/node@18/bin:$PATH"' >> ~/.zshrc
    source ~/.zshrc

elif [[ "$OS" == "ubuntu" ]]; then
    # Update package lists
    sudo apt-get update

    # Install system packages
    print_step "Installing system packages via apt..."
    sudo apt-get install -y \
        git \
        curl \
        ffmpeg \
        build-essential \
        cmake \
        pkg-config \
        libportaudio2 \
        libportaudiocpp0 \
        portaudio19-dev \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        nodejs \
        npm \
        wget \
        unzip \
        jq \
        libsndfile1 \
        libsndfile1-dev \
        libasound2-dev \
        libssl-dev \
        libffi-dev \
        libopenblas-dev \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        gfortran

    # Install Node.js 18.x
    print_step "Installing Node.js 18.x..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs

    # Install Rust
    print_step "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Install pyenv
print_step "Installing pyenv..."
curl https://pyenv.run | bash

# Add pyenv to shell configuration
if [[ "$OS" == "macos" ]]; then
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(pyenv init -)"' >> ~/.zshrc
    source ~/.zshrc
else
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    source ~/.bashrc
fi

# Install Python 3.11 via pyenv
print_step "Installing Python 3.11 via pyenv..."
pyenv install 3.11.7
pyenv global 3.11.7

# Clone repositories
print_step "Cloning required repositories..."
cd ~/sentient-avatar/src

# Clone main repositories
repos=(
    "vllm-project/vllm"
    "ggml-org/llama.cpp"
    "crewAIInc/crewAI"
    "microsoft/autogen"
    "langchain-ai/langchain"
    "run-llama/llama_index"
    "qdrant/qdrant"
    "ggml-org/whisper.cpp"
    "LLaVA-VL/LLaVA-NeXT"
    "suno-ai/bark"
    "coqui-ai/TTS"
    "rhasspy/piper"
    "OpenTalker/SadTalker"
    "Rudrabha/Wav2Lip"
    "SillyTavern/SillyTavern"
)

for repo in "${repos[@]}"; do
    repo_name=$(basename "$repo")
    if [ ! -d "$repo_name" ]; then
        print_step "Cloning $repo..."
        git clone "https://github.com/$repo.git"
    else
        print_warning "$repo_name already exists, skipping..."
    fi
done

# Create Python virtual environments
print_step "Creating Python virtual environments..."

# Function to create venv and install dependencies
create_venv() {
    local name=$1
    local requirements=$2
    
    print_step "Setting up $name virtual environment..."
    python -m venv "venv_$name"
    source "venv_$name/bin/activate"
    pip install --upgrade pip setuptools wheel
    if [ -f "$requirements" ]; then
        pip install -r "$requirements"
    fi
    deactivate
}

# Create venvs for major components
cd ~/sentient-avatar/src

# LLM components
create_venv "vllm" "vllm/requirements.txt"
create_venv "llama" "llama.cpp/requirements.txt"

# Agent components
create_venv "crewai" "crewAI/requirements.txt"
create_venv "autogen" "autogen/requirements.txt"
create_venv "langchain" "langchain/requirements.txt"
create_venv "llamaindex" "llama_index/requirements.txt"

# Audio components
create_venv "whisper" "whisper.cpp/requirements.txt"
create_venv "bark" "bark/requirements.txt"
create_venv "tts" "TTS/requirements.txt"
create_venv "piper" "piper/requirements.txt"

# Vision components
create_venv "llava" "LLaVA-NeXT/requirements.txt"
create_venv "sadtalker" "SadTalker/requirements.txt"
create_venv "wav2lip" "Wav2Lip/requirements.txt"

# Create .env file if it doesn't exist
if [ ! -f ~/sentient-avatar/.env ]; then
    print_step "Creating .env file..."
    cat > ~/sentient-avatar/.env << EOL
# Environment variables for Sentient Avatar
PYTHONPATH=\${PYTHONPATH}:~/sentient-avatar/src
STORAGE_DIR=~/sentient-avatar/storage
MODELS_DIR=~/sentient-avatar/models
LOGS_DIR=~/sentient-avatar/logs

# Optional: HuggingFace token for model downloads
# HUGGINGFACE_TOKEN=your_token_here

# Optional: GPU settings
# CUDA_VISIBLE_DEVICES=0
# MPS_DEVICE=0  # For Apple Silicon
EOL
fi

print_step "Bootstrap complete! Next steps:"
echo "1. Run './get-models.sh' to download required models"
echo "2. Review and adjust environment variables in .env"
echo "3. Start the system with 'make up' or 'docker-compose up'" 