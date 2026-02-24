"""Pytest fixtures for integration tests with Docker services."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Generator

import pytest
import requests


def wait_for_service(url: str, service_name: str, max_retries: int = 30, timeout: int = 2) -> bool:
    """Wait for a service to be healthy."""
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                print(f"✅ {service_name} is ready")
                return True
        except requests.exceptions.RequestException:
            if i % 5 == 0:  # Print every 5 attempts
                print(f"⏳ Waiting for {service_name}... ({i}/{max_retries})")
            time.sleep(2)
    
    print(f"❌ {service_name} failed to start after {max_retries * timeout}s")
    return False


@pytest.fixture(scope="session")
def docker_compose_file() -> Path:
    """Get path to docker-compose file for all services."""
    return Path(__file__).parent.parent.parent / "docker" / "all" / "docker-compose.yml"


@pytest.fixture(scope="session")
def docker_services(docker_compose_file: Path) -> Generator[None, None, None]:
    """Start docker services for integration tests.
    
    This fixture starts all services (Ollama, Qdrant, Milvus) before tests
    and tears them down after all tests are complete.
    """
    # Check if services are already running
    already_running = False
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True
        )
        running_containers = result.stdout.strip().split("\n")
        if "ollama" in running_containers and "qdrant" in running_containers:
            print("ℹ️  Docker services already running, skipping start")
            already_running = True
    except subprocess.CalledProcessError:
        pass
    
    if not already_running:
        print(f"🚀 Starting Docker services from {docker_compose_file}")
        
        # Start services
        subprocess.run(
            ["docker-compose", "-f", str(docker_compose_file), "up", "-d"],
            check=True,
            capture_output=True
        )
        
        print("⏳ Waiting for services to be healthy...")
        
        # Wait for services
        services = {
            "Ollama": "http://localhost:11434/api/tags",
            "Qdrant": "http://localhost:6333/healthz",
            "Milvus": "http://localhost:9091/healthz",
        }
        
        all_healthy = True
        for service_name, url in services.items():
            if not wait_for_service(url, service_name):
                all_healthy = False
        
        if not all_healthy:
            # Cleanup on failure
            subprocess.run(
                ["docker-compose", "-f", str(docker_compose_file), "down"],
                capture_output=True
            )
            pytest.fail("Failed to start all Docker services")
        
        print("✅ All services are ready for testing")
    
    yield
    
    # Cleanup (only if we started them)
    if not already_running:
        print("🧹 Stopping Docker services...")
        subprocess.run(
            ["docker-compose", "-f", str(docker_compose_file), "down"],
            capture_output=True
        )
        print("✅ Docker services stopped")


@pytest.fixture
def ollama_client(docker_services):
    """Provide Ollama client for integration tests."""
    from quaerium.infra.embedding.ollama import OllamaEmbeddingClient
    
    return OllamaEmbeddingClient(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )


@pytest.fixture
def qdrant_service(docker_services):
    """Provide Qdrant service for integration tests."""
    from quaerium.infra.vectorstores.qdrant import QdrantService
    from quaerium.infra.vectorstores.qdrant.config import QdrantConfig
    
    config = QdrantConfig(url="http://localhost:6333")
    return QdrantService(config)


@pytest.fixture
def milvus_service(docker_services):
    """Provide Milvus service for integration tests."""
    from quaerium.infra.vectorstores.factory import create_milvus_service
    
    return create_milvus_service(uri="http://localhost:19530")


@pytest.fixture
def test_collection_name() -> str:
    """Generate unique collection name for tests."""
    import uuid
    return f"test_{uuid.uuid4().hex[:8]}"
