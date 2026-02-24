"""Example: How to extend the unified test suite for a new vector store.

This file demonstrates how to add a new vector store to the unified testing framework.
"""

from __future__ import annotations

from typing import Any, Dict, List, Generator
import pytest

# Import base classes
from tests.test_infra.test_vectorstores.base_tests import (
    VectorStoreCollectionTests,
    VectorStoreDataTests,
    VectorStoreHealthTests,
)

# Import your new vector store (example)
# from quaerium.infra.vectorstores.newstore.config import NewStoreConfig
# from quaerium.infra.vectorstores.newstore.service import NewStoreService


# ============================================================================
# Step 1: Create Fixtures for Your Vector Store
# ============================================================================

@pytest.fixture(scope="session")
def newstore_service():  # -> Generator[NewStoreService, None, None]:
    """Provide NewStore service for testing.
    
    This fixture should:
    1. Create a config from environment variables
    2. Initialize the service
    3. Yield the service for tests
    4. Clean up after tests complete
    """
    # Example implementation:
    # config = NewStoreConfig(
    #     host=os.getenv("NEWSTORE_HOST", "localhost"),
    #     port=int(os.getenv("NEWSTORE_PORT", "9999")),
    # )
    # 
    # service = NewStoreService(config)
    # 
    # try:
    #     yield service
    # finally:
    #     service.close()
    
    pytest.skip("NewStore not implemented - this is an example")


@pytest.fixture
def newstore_collection_name() -> str:
    """Provide collection name for NewStore tests."""
    return "test_newstore_collection"


@pytest.fixture
def newstore_vector_size() -> int:
    """Provide vector size for NewStore tests."""
    return 384


# ============================================================================
# Step 2: Implement Concrete Test Classes
# ============================================================================

class TestNewStoreCollections(VectorStoreCollectionTests):
    """Test collection operations for NewStore.
    
    This class inherits standard tests from VectorStoreCollectionTests
    and implements the abstract fixtures.
    """
    
    @pytest.fixture
    def vector_store_service(self, newstore_service):
        """Provide NewStore service."""
        return newstore_service
    
    @pytest.fixture
    def collection_name(self, newstore_collection_name: str) -> str:
        """Provide collection name."""
        return newstore_collection_name
    
    @pytest.fixture
    def vector_size(self, newstore_vector_size: int) -> int:
        """Provide vector size."""
        return newstore_vector_size
    
    # Implement abstract test methods
    def test_create_collection(
        self,
        vector_store_service,  # : NewStoreService,
        collection_name: str,
        vector_size: int,
    ) -> None:
        """Test creating a collection in NewStore."""
        # Example implementation:
        # vector_store_service.create_collection(
        #     name=collection_name,
        #     dimension=vector_size,
        # )
        # 
        # assert vector_store_service.collection_exists(collection_name)
        # 
        # # Cleanup
        # vector_store_service.drop_collection(collection_name)
        pass
    
    def test_collection_exists(
        self,
        vector_store_service,
        collection_name: str,
    ) -> None:
        """Test checking if collection exists."""
        # Create collection
        # vector_store_service.create_collection(collection_name, dimension=384)
        # 
        # # Check it exists
        # assert vector_store_service.collection_exists(collection_name) is True
        # 
        # # Drop it
        # vector_store_service.drop_collection(collection_name)
        # 
        # # Check it doesn't exist
        # assert vector_store_service.collection_exists(collection_name) is False
        pass
    
    def test_drop_collection(
        self,
        vector_store_service,
        collection_name: str,
    ) -> None:
        """Test dropping a collection."""
        pass
    
    def test_list_collections(
        self,
        vector_store_service,
    ) -> None:
        """Test listing collections."""
        # collections = vector_store_service.list_collections()
        # assert isinstance(collections, list)
        pass


class TestNewStoreData(VectorStoreDataTests):
    """Test data operations for NewStore."""
    
    @pytest.fixture
    def vector_store_service(self, newstore_service):
        """Provide NewStore service."""
        return newstore_service
    
    @pytest.fixture
    def collection_name(self, newstore_collection_name: str) -> str:
        """Provide collection name."""
        return newstore_collection_name
    
    @pytest.fixture
    def vector_size(self, newstore_vector_size: int) -> int:
        """Provide vector size."""
        return newstore_vector_size
    
    def test_add_vectors(
        self,
        vector_store_service,
        collection_name: str,
        sample_ids: List[str],
        sample_vectors: List[List[float]],
        sample_metadata: List[Dict[str, Any]],
    ) -> None:
        """Test adding vectors."""
        # Setup collection
        # vector_store_service.create_collection(collection_name, dimension=len(sample_vectors[0]))
        # 
        # # Add vectors
        # vector_store_service.add(
        #     collection_name=collection_name,
        #     ids=sample_ids,
        #     vectors=sample_vectors,
        #     metadata=sample_metadata,
        # )
        # 
        # # Verify count
        # count = vector_store_service.count(collection_name)
        # assert count == len(sample_ids)
        # 
        # # Cleanup
        # vector_store_service.drop_collection(collection_name)
        pass
    
    def test_search_vectors(
        self,
        vector_store_service,
        collection_name: str,
        sample_vectors: List[List[float]],
    ) -> None:
        """Test searching vectors."""
        pass
    
    # Implement other abstract methods...
    def test_upsert_vectors(self, vector_store_service, collection_name, sample_ids, sample_vectors):
        pass
    
    def test_get_by_ids(self, vector_store_service, collection_name, sample_ids):
        pass
    
    def test_delete_vectors(self, vector_store_service, collection_name, sample_ids):
        pass
    
    def test_count_vectors(self, vector_store_service, collection_name):
        pass


class TestNewStoreHealth(VectorStoreHealthTests):
    """Test health checks for NewStore."""
    
    @pytest.fixture
    def vector_store_service(self, newstore_service):
        """Provide NewStore service."""
        return newstore_service
    
    @pytest.fixture
    def collection_name(self, newstore_collection_name: str) -> str:
        """Provide collection name."""
        return newstore_collection_name
    
    @pytest.fixture
    def vector_size(self, newstore_vector_size: int) -> int:
        """Provide vector size."""
        return newstore_vector_size
    
    def test_connection(self, vector_store_service):
        """Test connection is established."""
        # assert vector_store_service.connection.is_connected()
        pass


# ============================================================================
# Step 3: Add to Parameterized Tests (test_unified.py)
# ============================================================================

# In test_unified.py, add "newstore" to the parametrize decorator:
#
# @pytest.mark.parametrize("vector_store_name", ["milvus", "qdrant", "chroma", "newstore"])
# class TestUnifiedVectorStoreCollections:
#     def test_create_and_drop_collection(self, vector_store_name: str, request):
#         service = request.getfixturevalue(f"{vector_store_name}_service")
#         
#         # Add NewStore-specific logic
#         elif vector_store_name == "newstore":
#             service.create_collection("test_collection", dimension=384)
#         
#         # Test logic...


# ============================================================================
# Step 4: Update conftest.py
# ============================================================================

# Add the newstore_service fixture to tests/test_infra/test_vectorstores/conftest.py:
#
# @pytest.fixture(scope="session")
# def newstore_service() -> Generator[NewStoreService, None, None]:
#     config = NewStoreConfig(
#         host=os.getenv("NEWSTORE_HOST", "localhost"),
#         port=int(os.getenv("NEWSTORE_PORT", "9999")),
#     )
#     
#     service = NewStoreService(config)
#     
#     try:
#         yield service
#     finally:
#         service.close()


# ============================================================================
# Step 5: Run Tests
# ============================================================================

# Run all unified tests including NewStore:
#   pytest tests/test_infra/test_vectorstores/test_unified.py
#
# Run only NewStore tests:
#   pytest tests/test_infra/test_vectorstores/test_unified.py -k newstore
#
# Run NewStore-specific test classes:
#   pytest tests/test_infra/test_vectorstores/test_example_newstore.py


if __name__ == "__main__":
    print(__doc__)
