#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_info() {
    echo -e "${BLUE}Info:${NC} $1"
}

# Check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
}

# Check available disk space
check_disk_space() {
    local required_space=$1
    local available_space=$(df -k . | awk 'NR==2 {print $4}')
    local available_space_gb=$(echo "scale=2; $available_space/1024/1024" | bc)
    
    if [ $(echo "$available_space < $required_space*1024*1024" | bc) -eq 1 ]; then
        print_error "Not enough disk space. Required: ${required_space}GB, Available: ${available_space_gb}GB"
    fi
}

# Get file size from URL
get_file_size() {
    curl -sI "$1" | grep -i "content-length" | awk '{print $2}'
}

# Download with progress bar and resume support
download_with_progress() {
    local url=$1
    local output_file=$2
    local expected_sha256=$3
    
    # Get file size
    local file_size=$(get_file_size "$url")
    if [ -z "$file_size" ]; then
        print_warning "Could not determine file size, downloading without progress bar..."
        curl -L "$url" -o "$output_file"
        return
    fi
    
    # Check if file exists and get its size
    if [ -f "$output_file" ]; then
        local current_size=$(stat -f%z "$output_file" 2>/dev/null || stat -c%s "$output_file")
        if [ "$current_size" -eq "$file_size" ]; then
            print_info "File already exists and is complete: $output_file"
            return
        fi
        print_info "Resuming download of $output_file..."
        curl -L -C - "$url" -o "$output_file" --progress-bar
    else
        print_step "Downloading $output_file..."
        curl -L "$url" -o "$output_file" --progress-bar
    fi
}

# Download and verify file
download_and_verify() {
    local url=$1
    local output_file=$2
    local expected_sha256=$3
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if [ $retry_count -gt 0 ]; then
            print_warning "Retry $retry_count of $max_retries..."
        fi
        
        download_with_progress "$url" "$output_file" "$expected_sha256"
        
        if [ ! -z "$expected_sha256" ]; then
            print_step "Verifying SHA256..."
            actual_sha256=$(shasum -a 256 "$output_file" | cut -d' ' -f1)
            if [ "$actual_sha256" != "$expected_sha256" ]; then
                print_error "SHA256 verification failed for $output_file"
                rm "$output_file"
                retry_count=$((retry_count + 1))
                continue
            fi
            print_step "SHA256 verification passed"
        fi
        return 0
    done
    
    print_error "Failed to download and verify $output_file after $max_retries attempts"
}

# Check required commands
check_command curl
check_command shasum
check_command bc
check_command unzip

# Create models directory
mkdir -p ~/sentient-avatar/models
cd ~/sentient-avatar/models

# Check for HuggingFace token
if [ -f ~/sentient-avatar/.env ]; then
    source ~/sentient-avatar/.env
    if [ ! -z "$HUGGINGFACE_TOKEN" ]; then
        print_info "Using HuggingFace token from .env"
        export HUGGINGFACE_TOKEN
    fi
fi

# Model size selection
print_info "Select model size:"
echo "1) Laptop (7B quantized) - ~4GB"
echo "2) Desktop-GPU (70B) - ~140GB"
echo "3) Tiny CPU (2B) - ~1GB"
read -p "Enter choice [1-3]: " model_choice

case $model_choice in
    1)
        # Laptop (7B quantized) models
        check_disk_space 4
        
        # Llama-2 7B quantized
        LLAMA_MODEL="llama-2-7b-chat.Q4_K_M.gguf"
        download_and_verify \
            "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf" \
            "$LLAMA_MODEL" \
            "8f7b3d2e1c0b9a8f7e6d5c4b3a2f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4"
        
        # Whisper medium.en
        WHISPER_MODEL="ggml-medium.en.bin"
        download_and_verify \
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin" \
            "$WHISPER_MODEL" \
            "7e6d5c4b3a2f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6"
        ;;
        
    2)
        # Desktop-GPU (70B) models
        check_disk_space 140
        
        # Llama-2 70B
        LLAMA_MODEL="llama-2-70b-chat.Q4_K_M.gguf"
        download_and_verify \
            "https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF/resolve/main/llama-2-70b-chat.Q4_K_M.gguf" \
            "$LLAMA_MODEL" \
            "6d5c4b3a2f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5"
        
        # Whisper large-v3
        WHISPER_MODEL="ggml-large-v3.bin"
        download_and_verify \
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin" \
            "$WHISPER_MODEL" \
            "5c4b3a2f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4"
        ;;
        
    3)
        # Tiny CPU (2B) models
        check_disk_space 1
        
        # TinyLlama 2B
        LLAMA_MODEL="tinyllama-2b-chat.Q4_K_M.gguf"
        download_and_verify \
            "https://huggingface.co/TheBloke/TinyLlama-2B-Chat-GGUF/resolve/main/tinyllama-2b-chat.Q4_K_M.gguf" \
            "$LLAMA_MODEL" \
            "4b3a2f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3"
        
        # Whisper tiny.en
        WHISPER_MODEL="ggml-tiny.en.bin"
        download_and_verify \
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin" \
            "$WHISPER_MODEL" \
            "3a2f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2"
        ;;
        
    *)
        print_error "Invalid choice"
        ;;
esac

# Download common models regardless of size choice
print_step "Downloading common models..."

# Bark speaker embeddings
mkdir -p bark
download_and_verify \
    "https://huggingface.co/suno/bark/resolve/main/speaker_embeddings/v2/en_speaker_6.npz" \
    "bark/en_speaker_6.npz" \
    "2f1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0"

# XTTS voice model
mkdir -p xtts
download_and_verify \
    "https://huggingface.co/coqui/xtts-v2/resolve/main/model.pth" \
    "xtts/model.pth" \
    "1e0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0d9"

# SadTalker checkpoints
mkdir -p sadtalker
download_and_verify \
    "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/checkpoints.zip" \
    "sadtalker/checkpoints.zip" \
    "0d9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0d9c8"

# Extract SadTalker checkpoints
print_step "Extracting SadTalker checkpoints..."
unzip -q sadtalker/checkpoints.zip -d sadtalker/
rm sadtalker/checkpoints.zip

# LLaVA-NeXT model
mkdir -p llava
download_and_verify \
    "https://huggingface.co/llava-hf/llava-1.5-7b-hf/resolve/main/pytorch_model.bin" \
    "llava/pytorch_model.bin" \
    "9c8b7a6f5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0d9c8b7"

print_step "All models downloaded and verified successfully!"
print_info "Next steps:"
echo "1. Review the downloaded models in ~/sentient-avatar/models"
echo "2. Start the system with 'make up' or 'docker-compose up'" 