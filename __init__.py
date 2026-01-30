"""
Databricks Multi-Agent Prototype
================================

This prototype demonstrates how to replicate Azure AI Foundry's MACAE
(Multi-Agent Custom Automation Engine) using Databricks Mosaic AI.

Created for Ryder AI Engineer interview preparation.

Modules:
- agents/: Agent templates and implementations
- orchestration/: Supervisor orchestration patterns
- tools/: RAG and utility tools
- notebooks/: Demonstration notebooks

Key Equivalences:
- FoundryAgentTemplate → DatabricksAgentTemplate (LangGraph + ResponsesAgent)
- MagenticBuilder → StateGraph supervisor pattern
- Azure AI Search → Databricks Vector Search
- HumanApprovalMagenticManager → LangGraph interrupt()
"""

__version__ = "0.1.0"
