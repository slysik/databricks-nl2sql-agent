"""Agent implementations for Databricks multi-agent system."""

# Simple NL2SQL agent (no newer MLflow dependency)
from agents.sql_agent import DatabricksNL2SQLAgent

__all__ = [
    "DatabricksNL2SQLAgent",
]

# Advanced agents require MLflow 2.16+ with ResponsesAgent support
# Uncomment when running on Databricks with compatible MLflow:
# from databricks_prototype.agents.base_agent import (
#     DatabricksAgentTemplate,
#     create_databricks_agent,
# )
