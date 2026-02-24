"""
Utility functions for Neo4j type conversions and helpers.
"""

from typing import Any

from neo4j.time import DateTime, Date, Time


def convert_neo4j_types(obj: Any) -> Any:
    """
    Recursively convert Neo4j types to JSON-serializable Python types.

    Converts:
    - neo4j.time.DateTime → ISO 8601 string
    - neo4j.time.Date → ISO 8601 date string
    - neo4j.time.Time → ISO 8601 time string
    - dict → recursively convert values
    - list → recursively convert items

    Args:
        obj: Object to convert (can be dict, list, Neo4j type, or primitive)

    Returns:
        JSON-serializable version of the object

    Example:
        >>> from neo4j.time import DateTime
        >>> dt = DateTime(2024, 1, 15, 10, 30, 0)
        >>> convert_neo4j_types(dt)
        '2024-01-15T10:30:00Z'
        >>> convert_neo4j_types({"created": dt, "count": 42})
        {'created': '2024-01-15T10:30:00Z', 'count': 42}
    """
    if isinstance(obj, DateTime):
        # Convert Neo4j DateTime to ISO string
        return obj.iso_format()
    elif isinstance(obj, Date):
        # Convert Neo4j Date to ISO string (YYYY-MM-DD)
        return obj.iso_format()
    elif isinstance(obj, Time):
        # Convert Neo4j Time to ISO string (HH:MM:SS)
        return obj.iso_format()
    elif isinstance(obj, dict):
        # Recursively convert dict values
        return {k: convert_neo4j_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively convert list/tuple items
        return [convert_neo4j_types(item) for item in obj]
    else:
        # Return as-is (int, float, str, bool, None)
        return obj


__all__ = ["convert_neo4j_types"]
