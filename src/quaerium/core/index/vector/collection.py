"""Collection management utilities for Milvus."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .connection import MilvusConnectionManager
from .exceptions import CollectionError

try:
    from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
except ImportError as exc:  # pragma: no cover
    MilvusClient = None  # type: ignore
    _pymilvus_import_error = exc


class MilvusCollectionManager:
    """Create, drop, and manage Milvus collections using MilvusClient."""

    def __init__(self, connection: MilvusConnectionManager) -> None:
        if MilvusClient is None:
            raise ImportError("pymilvus is required for Milvus operations") from _pymilvus_import_error
        self.connection = connection

    @property
    def client(self) -> MilvusClient:
        return self.connection.client

    def ensure_collection(
        self,
        name: str,
        schema: Any,
        *,
        index_params: Optional[Dict[str, Any]] = None,
        load: bool = True,
        shards_num: int = 2,
    ) -> None:
        """Create the collection if missing; optionally create index and load."""
        self.connection.ensure()
        try:
            if isinstance(schema, dict):
                # Convert schema dict to CollectionSchema
                fields = []
                vector_field_name = None
                for key, value in schema.items():
                    dtype_value = value.get("dtype", "FLOAT_VECTOR")
                    
                    # Convert string dtype to DataType enum
                    if isinstance(dtype_value, str):
                        dtype = getattr(DataType, dtype_value, DataType.FLOAT_VECTOR)
                    else:
                        dtype = dtype_value
                    
                    # Track vector field for default index
                    if dtype_value == "FLOAT_VECTOR":
                        vector_field_name = key
                    
                    field_kwargs = {k: v for k, v in value.items() if k != "dtype"}
                    fields.append(FieldSchema(name=key, dtype=dtype, **field_kwargs))
                schema = CollectionSchema(fields=fields)

            if not self.client.has_collection(collection_name=name):
                self.client.create_collection(
                    collection_name=name,
                    schema=schema,
                    shards_num=shards_num,
                )
            
            # Create index if specified or create default index for vector field
            if index_params:
                field_name = index_params.get("field_name", "embedding")
                existing = self.client.list_indexes(collection_name=name)
                # list_indexes may return list[str] or list[dict]
                has_index = any(
                    (idx.get("field_name") if isinstance(idx, dict) else None) == field_name or idx == field_name
                    for idx in (existing or [])
                )
                if not has_index:
                    if hasattr(index_params, "to_dict"):
                        idx_params = index_params  # assume already IndexParams
                    else:
                        idx = self.client.prepare_index_params()
                        idx.add_index(
                            field_name=field_name,
                            index_type=index_params.get("index_type", "HNSW"),
                            metric_type=index_params.get("metric_type", "IP"),
                            params={k: v for k, v in index_params.items() if k not in {"field_name", "index_type", "metric_type"}},
                        )
                        idx_params = idx
                    self.client.create_index(collection_name=name, index_params=idx_params)
            elif vector_field_name:
                # Create default index for vector field if no index params provided
                existing = self.client.list_indexes(collection_name=name)
                has_index = any(
                    (idx.get("field_name") if isinstance(idx, dict) else None) == vector_field_name or idx == vector_field_name
                    for idx in (existing or [])
                )
                if not has_index:
                    idx = self.client.prepare_index_params()
                    idx.add_index(
                        field_name=vector_field_name,
                        index_type="HNSW",
                        metric_type="IP",
                    )
                    self.client.create_index(collection_name=name, index_params=idx)
            
            if load:
                self.client.load_collection(collection_name=name)
        except Exception as exc:  # pragma: no cover - passthrough
            raise CollectionError(f"Failed to ensure collection '{name}'") from exc

    def drop_collection(self, name: str) -> None:
        """Drop a collection if it exists."""
        self.connection.ensure()
        try:
            if self.client.has_collection(collection_name=name):
                self.client.drop_collection(collection_name=name)
        except Exception as exc:  # pragma: no cover - passthrough
            raise CollectionError(f"Failed to drop collection '{name}'") from exc

    def load(self, name: str) -> None:
        """Load a collection into memory."""
        self.connection.ensure()
        try:
            self.client.load_collection(collection_name=name)
        except Exception as exc:  # pragma: no cover - passthrough
            raise CollectionError(f"Failed to load collection '{name}'") from exc

    def release(self, name: str) -> None:
        """Release a collection from memory."""
        self.connection.ensure()
        try:
            self.client.release_collection(collection_name=name)
        except Exception as exc:  # pragma: no cover - passthrough
            raise CollectionError(f"Failed to release collection '{name}'") from exc

    def create_index(self, name: str, field_name: str, index_params: Dict[str, Any]) -> None:
        """Create an index on a field."""
        self.connection.ensure()
        try:
            self.client.create_index(collection_name=name, field_name=field_name, index_params=index_params)
        except Exception as exc:  # pragma: no cover - passthrough
            raise CollectionError(f"Failed to create index on '{field_name}' for '{name}'") from exc


__all__ = ["MilvusCollectionManager"]
