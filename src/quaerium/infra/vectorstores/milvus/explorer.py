from __future__ import annotations

from typing import Any, Dict, List, Optional

from quaerium.infra.vectorstores.milvus.connection import MilvusConnectionManager
from quaerium.infra.vectorstores.milvus.data import MilvusDataManager
from quaerium.infra.vectorstores.milvus.exceptions import CollectionError


class MilvusExplorer:
    """Utility class for inspecting Milvus collections and previewing data."""

    def __init__(self, connection: MilvusConnectionManager) -> None:
        self.connection = connection
        self.data = MilvusDataManager(connection)

    def list_collections(self) -> List[Dict[str, Any]]:
        """List collections with basic metadata."""
        self.connection.ensure()
        colls = self.connection.client.list_collections()
        results: List[Dict[str, Any]] = []
        for name in colls:
            try:
                desc = self.connection.client.describe_collection(collection_name=name)
                stats = self.connection.client.get_collection_stats(collection_name=name)
                results.append(
                    {
                        "name": name,
                        "description": desc.get("description"),
                        "row_count": stats.get("row_count"),
                    }
                )
            except Exception:
                results.append({"name": name, "description": None, "row_count": None})
        return results

    def collection_schema(self, name: str) -> Dict[str, Any]:
        """Return schema details for a collection."""
        self.connection.ensure()
        try:
            desc = self.connection.client.describe_collection(collection_name=name)
            return desc
        except Exception as exc:  # pragma: no cover
            raise CollectionError(f"Failed to read schema for collection '{name}'") from exc

    def _build_default_expr(self, desc: Dict[str, Any]) -> str:
        """Choose a permissive expr based on primary key type."""
        pk = None
        for f in desc.get("fields", []):
            if f.get("is_primary"):
                pk = f
                break
        if pk is None:
            return "1 == 1"
        dtype_str = str(pk.get("type", "")).lower()
        name = pk.get("name", "id")
        if "varchar" in dtype_str or "string" in dtype_str:
            return f"{name} != ''"
        if "int" in dtype_str or "float" in dtype_str or "double" in dtype_str:
            return f"{name} >= 0"
        # default safe expr for unknown types
        return f"{name} != ''"

    def preview(self, name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the first N entities from a collection (excluding vector fields)."""
        self.connection.ensure()
        desc = self.connection.client.describe_collection(collection_name=name)
        expr = self._build_default_expr(desc)
        # choose non-vector fields for readability
        output_fields: List[str] = []
        for f in desc.get("fields", []):
            dtype_str = str(f.get("type", "")).lower()
            if f.get("name") in {"embedding", "id"}:
                continue
            if "float_vector" in dtype_str or "binary_vector" in dtype_str:
                continue
            output_fields.append(f.get("name"))
        try:
            results = self.data.query(collection_name=name, expr=expr, limit=limit, output_fields=output_fields)
            return results or []
        except Exception:
            # fallback: try without expr
            try:
                results = self.data.query(collection_name=name, expr="1 == 1", limit=limit, output_fields=output_fields)
                return results or []
            except Exception as exc:  # pragma: no cover
                raise CollectionError(f"Failed to preview collection '{name}': {exc}") from exc

    def get_collection_data(
        self,
        name: str,
        *,
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Fetch up to `limit` rows from a collection (all fields unless specified).

        This is a simple helper for UI/debug; it loads the collection and runs a query
        with a permissive expression.
        """
        self.connection.ensure()
        desc = self.connection.client.describe_collection(collection_name=name)
        expr = self._build_default_expr(desc)
        try:
            results = self.data.query(collection_name=name, expr=expr, limit=limit, output_fields=output_fields)
            return results or []
        except Exception:
            try:
                results = self.data.query(collection_name=name, expr="1 == 1", limit=limit, output_fields=output_fields)
                return results or []
            except Exception as exc:  # pragma: no cover
                raise CollectionError(f"Failed to read collection '{name}': {exc}") from exc


__all__ = ["MilvusExplorer"]
