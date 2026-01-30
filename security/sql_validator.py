"""
SQL Validator - Prevents SQL injection in NL2SQL systems.

INTERVIEW NOTE: LLM-generated SQL is untrusted input.
This is the last line of defense after Unity Catalog RBAC.
"""
from typing import Final
import sqlparse

class SecurityError(Exception):
    """Raised when SQL validation fails."""
    pass

# INTERVIEW NOTE: frozenset for O(1) lookup
BLOCKED_KEYWORDS: Final[frozenset[str]] = frozenset([
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE",
    "TRUNCATE", "GRANT", "REVOKE", "UNION", "--", ";"
])

MAX_ROWS: Final[int] = 1000

def validate_sql(sql: str) -> str:
    """Validate SQL before execution.

    INTERVIEW NOTE: Called before EVERY query execution.
    Defense-in-depth: RBAC → System Prompt → This Validator

    Args:
        sql: The SQL statement to validate

    Returns:
        Validated SQL (possibly with LIMIT added)

    Raises:
        SecurityError: If SQL contains blocked patterns
    """
    # Parse into AST (catches structural attacks)
    parsed = sqlparse.parse(sql)

    if len(parsed) != 1:
        raise SecurityError("Multiple statements blocked")

    if parsed[0].get_type() != "SELECT":
        raise SecurityError(f"Only SELECT allowed. Got: {parsed[0].get_type()}")

    # Check blocked keywords
    sql_upper = sql.upper()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in sql_upper:
            raise SecurityError(f"Blocked keyword: {keyword}")

    # Enforce LIMIT
    if "LIMIT" not in sql_upper:
        sql = f"{sql} LIMIT {MAX_ROWS}"

    return sql
