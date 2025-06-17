#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        exit 1
    fi
}

# Function to check disk space
check_disk_space() {
    local required_space=50 # GB
    local available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ "$available_space" -lt "$required_space" ]; then
        echo -e "${RED}Error: At least ${required_space}GB of free disk space is required${NC}"
        exit 1
    fi
}

# Function to check system requirements
check_system_requirements() {
    echo -e "${YELLOW}Checking system requirements...${NC}"
    
    # Check commands
    check_command docker
    check_command docker-compose
    check_command git
    check_command curl
    check_command jq
    
    # Check disk space
    check_disk_space
    
    # Check NVIDIA GPU if available
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}NVIDIA GPU detected${NC}"
    else
        echo -e "${YELLOW}Warning: No NVIDIA GPU detected. Some services may not work optimally${NC}"
    fi
}

# Function to generate certificates
generate_certificates() {
    echo -e "${YELLOW}Generating certificates...${NC}"
    
    # Create certificates directory
    mkdir -p config/certs
    
    # Generate CA certificate
    openssl genrsa -out config/certs/ca.key 4096
    openssl req -x509 -new -nodes -key config/certs/ca.key -sha256 -days 365 -out config/certs/ca.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=Sentient Avatar CA"
    
    # Generate service certificates
    for service in traefik vault consul; do
        openssl genrsa -out config/certs/$service.key 2048
        openssl req -new -key config/certs/$service.key -out config/certs/$service.csr \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=$service.sentient-avatar.local"
        openssl x509 -req -in config/certs/$service.csr -CA config/certs/ca.pem \
            -CAkey config/certs/ca.key -CAcreateserial -out config/certs/$service.pem -days 365 -sha256
    done
}

# Function to generate secrets
generate_secrets() {
    echo -e "${YELLOW}Generating secrets...${NC}"
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        cat > .env << EOF
# Database credentials
ELASTIC_PASSWORD=$(openssl rand -base64 32)
KIBANA_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 32)

# Service tokens
CONSUL_MASTER_TOKEN=$(openssl rand -base64 32)
CONSUL_AGENT_TOKEN=$(openssl rand -base64 32)
CONSUL_ENCRYPT_KEY=$(openssl rand -base64 32)
TRANSIT_TOKEN=$(openssl rand -base64 32)
QDRANT_API_KEY=$(openssl rand -base64 32)

# Model paths
LLAMA_MODEL=llama-2-7b-chat.gguf
WHISPER_MODEL=whisper-medium.en.gguf
XTTS_MODEL=xtts-v2
SADTALKER_MODEL=sadtalker
LLAVA_MODEL=llava-next
EOF
    fi
}

# Function to create directories
create_directories() {
    echo -e "${YELLOW}Creating directories...${NC}"
    
    # Create storage directories
    mkdir -p storage/{redis,rabbitmq,elasticsearch,qdrant,consul,vault,prometheus,grafana}
    
    # Create log directories
    mkdir -p logs/{api,llm,asr,tts,avatar,vision}
}

# Function to build and start services
start_services() {
    echo -e "${YELLOW}Building and starting services...${NC}"
    
    # Build images
    docker-compose build
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be healthy
    echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
    sleep 30
    
    # Check service health
    for service in api llm asr tts avatar vision; do
        if ! docker-compose ps $service | grep -q "healthy"; then
            echo -e "${RED}Error: $service service is not healthy${NC}"
            exit 1
        fi
    done
}

# Main deployment process
echo -e "${GREEN}Starting Sentient Avatar deployment...${NC}"

# Check system requirements
check_system_requirements

# Generate certificates and secrets
generate_certificates
generate_secrets

# Create directories
create_directories

# Start services
start_services

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Download models using ./get-models.sh"
echo "2. Access the UI at http://localhost:3000"
echo "3. Access the API at http://localhost:8005"
echo "4. Access monitoring at http://localhost:3000 (Grafana)"
 