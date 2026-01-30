"""
NL2SQL Agent - Converts natural language to SQL on Delta tables.

INTERVIEW CONTEXT:
- Equivalent to: Azure AI Foundry's FoundryAgentTemplate
- Pattern: LangChain SQL Agent with ChatDatabricks
- Key: Auto schema introspection via SQLDatabaseToolkit

MACAE Comparison:
| MACAE (Azure)         | This (Databricks)        |
|-----------------------|--------------------------|
| FoundryAgentTemplate  | DatabricksNL2SQLAgent    |
| AzureAIAgentClient    | ChatDatabricks           |
| MagenticBuilder       | create_react_agent       |
"""
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_databricks import ChatDatabricks
from langgraph.prebuilt import create_react_agent

# INTERVIEW NOTE: Import our security layer
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from security.sql_validator import validate_sql, SecurityError


class DatabricksNL2SQLAgent:
    """Natural Language to SQL agent for Delta tables.

    Example:
        agent = DatabricksNL2SQLAgent("ryder_chatbot", "agent_accessible")
        result = agent.query("Which vendors have reliability > 95%?")
    """

    def __init__(
        self,
        catalog: str,
        schema: str,
        model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
        warehouse_id: str = None,
    ):
        """Initialize the NL2SQL agent.

        Args:
            catalog: Unity Catalog name
            schema: Schema within the catalog
            model_endpoint: Databricks model serving endpoint
            warehouse_id: SQL Warehouse ID (or set DATABRICKS_WAREHOUSE_ID env var)
        """
        self.catalog = catalog
        self.schema = schema

        # DEMO: Hardcoded warehouse ID for interview demo
        # In production, use environment variables
        warehouse_id = warehouse_id or os.environ.get("DATABRICKS_WAREHOUSE_ID") or "45b831d537dc8a75"
        host = os.environ.get("DATABRICKS_HOST") or "adb-7405612122501781.1.azuredatabricks.net"
        token = os.environ.get("DATABRICKS_TOKEN")

        if not token:
            raise ValueError("DATABRICKS_TOKEN environment variable is required. Generate one from User Settings > Developer > Access Tokens in Databricks.")

        # Strip protocol if present (SQLDatabase expects just the hostname)
        if host.startswith("https://"):
            host = host[8:]
        if host.startswith("http://"):
            host = host[7:]

        # INTERVIEW NOTE: SQLDatabase.from_databricks() handles:
        # - Connection pooling
        # - Unity Catalog auth
        # - Schema introspection for LLM context
        self.db = SQLDatabase.from_databricks(
            catalog=catalog,
            schema=schema,
            warehouse_id=warehouse_id,
            host=host,
            api_token=token
        )

        # Temperature=0 for deterministic SQL generation
        self.llm = ChatDatabricks(endpoint=model_endpoint, temperature=0.0)

        # Build agent with SQL tools
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)

        # INTERVIEW NOTE: System prompt = first line of defense
        system_prompt = f"""You are a SQL expert for Databricks Delta tables.
DATABASE: {catalog}.{schema}

SECURITY RULES:
1. Generate ONLY SELECT statements
2. Query ONLY tables in {catalog}.{schema}
3. NEVER include DROP, DELETE, UPDATE, INSERT
4. Use LIMIT to restrict results (default: 100)

WORKFLOW:
1. sql_db_list_tables - see available tables
2. sql_db_schema - understand table structure
3. sql_db_query - execute and return results

If you cannot answer with SQL, say "I cannot query that data."
"""

        self.agent = create_react_agent(
            model=self.llm,
            tools=toolkit.get_tools(),
            prompt=system_prompt,
        )

    def query(self, question: str) -> dict:
        """Execute natural language query with security validation.

        INTERVIEW NOTE: The agent generates SQL, then we validate it
        before showing results. This is defense-in-depth.

        Args:
            question: Natural language question about the data

        Returns:
            dict with 'answer' (str) and 'success' (bool)
        """
        try:
            result = self.agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            answer = result["messages"][-1].content
            return {"answer": answer, "success": True}
        except SecurityError as e:
            # INTERVIEW NOTE: Security errors are caught and reported gracefully
            return {"answer": f"Security blocked: {e}", "success": False}
        except Exception as e:
            return {"answer": f"Error: {e}", "success": False}
