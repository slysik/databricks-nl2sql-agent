"""Security module for SQL validation and injection prevention."""
from security.sql_validator import validate_sql, SecurityError

__all__ = ["validate_sql", "SecurityError"]
