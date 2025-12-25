"""Generate synthetic vector data for benchmarks."""

from __future__ import annotations

import random
from typing import Any, Dict, List
from uuid import uuid4


class VectorDataGenerator:
    """Generate synthetic vector data for benchmarking."""
    
    def __init__(self, dimension: int = 384, seed: int = 42):
        """Initialize generator.
        
        Args:
            dimension: Vector dimensionality
            seed: Random seed for reproducibility
        """
        self.dimension = dimension
        self.seed = seed
        random.seed(seed)
    
    def generate_vectors(self, count: int) -> List[List[float]]:
        """Generate random vectors.
        
        Args:
            count: Number of vectors to generate
            
        Returns:
            List of vectors
        """
        return [
            [random.random() for _ in range(self.dimension)]
            for _ in range(count)
        ]
    
    def generate_vector(self) -> List[float]:
        """Generate a single random vector.
        
        Returns:
            Single vector
        """
        return [random.random() for _ in range(self.dimension)]
    
    def generate_data(
        self,
        count: int,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate data points with vectors and metadata.
        
        Args:
            count: Number of data points
            include_metadata: Include metadata fields
            
        Returns:
            List of data points
        """
        data = []
        categories = ["tech", "science", "business", "sports", "entertainment"]
        
        for i in range(count):
            point = {
                "id": str(uuid4()),
                "vector": self.generate_vector(),
            }
            
            if include_metadata:
                point["metadata"] = {
                    "index": i,
                    "category": categories[i % len(categories)],
                    "score": random.uniform(0, 100),
                    "text": f"Document {i} about {categories[i % len(categories)]}",
                }
            
            data.append(point)
        
        return data
    
    def generate_milvus_data(
        self,
        count: int,
        start_id: int = 1,
    ) -> List[Dict[str, Any]]:
        """Generate data in unified format with INT64 IDs for Milvus.
        
        Args:
            count: Number of data points
            start_id: Starting ID value
            
        Returns:
            List of data points
        """
        categories = ["tech", "science", "business", "sports", "entertainment"]
        
        return [
            {
                "id": start_id + i,
                "vector": self.generate_vector(),
                "text": f"Document {i} about {categories[i % len(categories)]}",
                "category": categories[i % len(categories)],
                "score": int(random.uniform(0, 100)),
            }
            for i in range(count)
        ]
    
    def generate_qdrant_points(
        self,
        count: int,
    ) -> List[Dict[str, Any]]:
        """Generate data in unified format with UUID IDs for Qdrant.
        
        Args:
            count: Number of points
            
        Returns:
            List of data points
        """
        categories = ["tech", "science", "business", "sports", "entertainment"]
        
        return [
            {
                "id": str(uuid4()),
                "vector": self.generate_vector(),
                "text": f"Document {i} about {categories[i % len(categories)]}",
                "category": categories[i % len(categories)],
                "score": int(random.uniform(0, 100)),
            }
            for i in range(count)
        ]
    
    def generate_chroma_data(
        self,
        count: int,
    ) -> List[Dict[str, Any]]:
        """Generate data in unified format with string IDs for ChromaDB.
        
        Args:
            count: Number of documents
            
        Returns:
            List of data points
        """
        categories = ["tech", "science", "business", "sports", "entertainment"]
        
        return [
            {
                "id": str(uuid4()),
                "vector": self.generate_vector(),
                "text": f"Document {i} about {categories[i % len(categories)]}",
                "category": categories[i % len(categories)],
                "score": int(random.uniform(0, 100)),
            }
            for i in range(count)
        ]


__all__ = ["VectorDataGenerator"]
