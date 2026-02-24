#!/usr/bin/env bash
# Docker helper script for RAG Toolkit services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

print_success() { print_msg "$GREEN" "✅ $@"; }
print_error() { print_msg "$RED" "❌ $@"; }
print_warning() { print_msg "$YELLOW" "⚠️  $@"; }
print_info() { print_msg "$BLUE" "ℹ️  $@"; }

# Show usage
usage() {
    cat << EOF
Usage: $0 <command> [service]

Commands:
    up [service]        Start services (all, milvus, qdrant, chroma, neo4j, ollama)
    down [service]      Stop services
    restart [service]   Restart services
    logs [service]      View logs
    ps                  Show running services
    clean [service]     Stop and remove volumes (deletes data!)
    health              Check health of all services
    pull-models         Pull Ollama models
    help                Show this help

Services:
    all                 All services (default)
    milvus              Milvus only
    qdrant              Qdrant only
    chroma              ChromaDB only
    neo4j               Neo4j only

Examples:
    $0 up                    # Start all services
    $0 up milvus             # Start only Milvus
    $0 logs qdrant           # View Qdrant logs
    $0 up chroma             # Start only ChromaDB
    $0 up neo4j              # Start only Neo4j
    $0 down                  # Stop all services
    $0 clean all             # Remove all data (careful!)
    $0 health                # Check service health
    $0 pull-models           # Download Ollama models

EOF
    exit 1
}

# Get docker-compose file path
get_compose_file() {
    local service=${1:-all}
    echo "$SCRIPT_DIR/$service/docker-compose.yml"
}

# Check if docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
}

# Start services
cmd_up() {
    local service=${1:-all}
    local compose_file=$(get_compose_file "$service")
    
    check_docker
    
    if [ ! -f "$compose_file" ]; then
        print_error "Service '$service' not found. Available: all, milvus, qdrant, chroma, neo4j"
        exit 1
    fi
    
    print_info "Starting $service services..."
    docker-compose -f "$compose_file" up -d
    
    print_info "Waiting for services to be healthy..."
    sleep 5
    
    cmd_health "$service"
    print_success "$service services started!"
    
    if [ "$service" = "all" ]; then
        print_warning "Don't forget to pull Ollama models: $0 pull-models"
    fi
}

# Stop services
cmd_down() {
    local service=${1:-all}
    local compose_file=$(get_compose_file "$service")
    
    if [ ! -f "$compose_file" ]; then
        print_error "Service '$service' not found"
        exit 1
    fi
    
    print_info "Stopping $service services..."
    docker-compose -f "$compose_file" down
    print_success "$service services stopped!"
}

# Restart services
cmd_restart() {
    local service=${1:-all}
    cmd_down "$service"
    sleep 2
    cmd_up "$service"
}

# View logs
cmd_logs() {
    local service=${1:-all}
    local compose_file=$(get_compose_file "$service")
    
    if [ ! -f "$compose_file" ]; then
        print_error "Service '$service' not found"
        exit 1
    fi
    
    docker-compose -f "$compose_file" logs -f
}

# Show running services
cmd_ps() {
    check_docker
    print_info "Running RAG Toolkit services:"
    docker ps --filter "name=milvus\|qdrant\|ollama\|chromadb\|neo4j" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# Clean (remove volumes)
cmd_clean() {
    local service=${1:-all}
    local compose_file=$(get_compose_file "$service")
    
    if [ ! -f "$compose_file" ]; then
        print_error "Service '$service' not found"
        exit 1
    fi
    
    print_warning "This will DELETE ALL DATA for $service services!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Aborted"
        exit 0
    fi
    
    print_info "Stopping and removing $service services and volumes..."
    docker-compose -f "$compose_file" down -v
    
    # Also remove local volume directories
    local volume_dir="$SCRIPT_DIR/$service/volumes"
    if [ -d "$volume_dir" ]; then
        print_info "Removing local volumes: $volume_dir"
        rm -rf "$volume_dir"
    fi
    
    print_success "$service services and data removed!"
}

# Check health
cmd_health() {
    local service=${1:-all}
    check_docker
    
    print_info "Checking health of services..."
    
    # Check Ollama
    if [ "$service" = "all" ] || [ "$service" = "ollama" ]; then
        if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_success "Ollama is healthy (http://localhost:11434)"
        else
            print_error "Ollama is not responding"
        fi
    fi
    
    # Check Qdrant
    if [ "$service" = "all" ] || [ "$service" = "qdrant" ]; then
        if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
            print_success "Qdrant is healthy (http://localhost:6333)"
        else
            print_error "Qdrant is not responding"
        fi
    fi
    
    # Check Milvus
    if [ "$service" = "all" ] || [ "$service" = "milvus" ]; then
        if curl -sf http://localhost:9091/healthz > /dev/null 2>&1; then
            print_success "Milvus is healthy (http://localhost:19530)"
        else
            print_error "Milvus is not responding"
        fi
    fi
    
    # Check ChromaDB
    if [ "$service" = "all" ] || [ "$service" = "chroma" ]; then
        if curl -sf http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
            print_success "ChromaDB is healthy (http://localhost:8000)"
        else
            print_error "ChromaDB is not responding"
        fi
    fi

    # Check Neo4j
    if [ "$service" = "all" ] || [ "$service" = "neo4j" ]; then
        if curl -sf http://localhost:7474 > /dev/null 2>&1; then
            print_success "Neo4j is healthy (Browser: http://localhost:7474, Bolt: bolt://localhost:7687)"
        else
            print_error "Neo4j is not responding"
        fi
    fi
}

# Pull Ollama models
cmd_pull_models() {
    check_docker
    
    if ! docker ps --format '{{.Names}}' | grep -q "^ollama$"; then
        print_error "Ollama is not running. Start it first: $0 up all"
        exit 1
    fi
    
    print_info "Pulling Ollama models..."
    
    # Embedding model
    print_info "Pulling nomic-embed-text (274MB)..."
    docker exec ollama ollama pull nomic-embed-text
    print_success "nomic-embed-text downloaded"
    
    # Optional: LLM
    read -p "Pull llama3.2 for chat? (~2GB) (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Pulling llama3.2..."
        docker exec ollama ollama pull llama3.2
        print_success "llama3.2 downloaded"
    fi
    
    print_info "Available models:"
    docker exec ollama ollama list
}

# Main
main() {
    if [ $# -eq 0 ]; then
        usage
    fi
    
    local command=$1
    shift
    
    case "$command" in
        up)
            cmd_up "$@"
            ;;
        down)
            cmd_down "$@"
            ;;
        restart)
            cmd_restart "$@"
            ;;
        logs)
            cmd_logs "$@"
            ;;
        ps)
            cmd_ps
            ;;
        clean)
            cmd_clean "$@"
            ;;
        health)
            cmd_health "$@"
            ;;
        pull-models)
            cmd_pull_models
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            print_error "Unknown command: $command"
            usage
            ;;
    esac
}

main "$@"
