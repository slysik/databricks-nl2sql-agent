"""
RAG Tools for Databricks Mosaic AI
Equivalent to Azure AI Search RAG in MACAE's FoundryAgentTemplate

This demonstrates Vector Search integration instead of Azure AI Search.
"""

from typing import Optional


def create_vector_search_tool(
    index_name: str,
    endpoint_name: str,
    description: str = "Search documents for relevant information",
    columns: Optional[list] = None,
    num_results: int = 5,
):
    """
    Create a RAG retriever tool using Databricks Vector Search.

    Equivalent to MACAE's Azure AI Search configuration:

    Azure AI Foundry (MACAE):
        search_config = SearchConfig(
            index_name="contract-summary-doc-index",
            connection_name="azure-ai-search-connection",
            search_query_type="hybrid"
        )

    Databricks (This):
        rag_tool = create_vector_search_tool(
            index_name="catalog.schema.contract_summary_index",
            endpoint_name="contract-vs-endpoint"
        )

    Args:
        index_name: Full Unity Catalog path (catalog.schema.index_name)
        endpoint_name: Vector Search endpoint name
        description: Tool description for agent
        columns: Columns to return (default: all)
        num_results: Number of results to retrieve

    Returns:
        LangChain retriever tool
    """
    try:
        from databricks.vector_search.client import VectorSearchClient
        from langchain_databricks import DatabricksVectorSearch
        from langchain.tools.retriever import create_retriever_tool

        # Initialize Vector Search client
        vsc = VectorSearchClient()

        # Create retriever
        retriever = DatabricksVectorSearch(
            index_name=index_name,
            endpoint=endpoint_name,
            text_column="content",
            columns=columns or ["doc_id", "title", "content", "metadata"],
        ).as_retriever(
            search_kwargs={"k": num_results}
        )

        # Wrap as tool
        return create_retriever_tool(
            retriever,
            name="search_documents",
            description=description
        )

    except ImportError:
        # Fallback for local development without Databricks
        return _create_mock_rag_tool(description)


def _create_mock_rag_tool(description: str):
    """Mock RAG tool for local development/testing"""
    from langchain_core.tools import tool

    @tool
    def search_documents(query: str) -> str:
        """Search documents for relevant information."""
        return f"[Mock RAG Result] Searched for: {query}\n" \
               f"Found relevant documents about the query topic."

    search_documents.description = description
    return search_documents


# ===== Supply Chain Specific Tools (Ryder Use Cases) =====

def create_demand_forecast_rag_tool(
    index_name: str = "supply_chain.forecasting.historical_data_index",
    endpoint_name: str = "demand-vs-endpoint"
):
    """
    RAG tool for demand forecasting historical data.

    Ryder Use Case: ML on historical data for inventory optimization

    This tool retrieves historical demand patterns to inform
    the Demand Forecasting Agent.
    """
    return create_vector_search_tool(
        index_name=index_name,
        endpoint_name=endpoint_name,
        description=(
            "Search historical demand data, seasonal patterns, and inventory "
            "trends to inform demand forecasting. Use this to find past demand "
            "patterns for specific products, regions, or time periods."
        ),
        columns=["date", "product_id", "region", "demand_value", "seasonality_factor"],
        num_results=10
    )


def create_route_data_rag_tool(
    index_name: str = "supply_chain.logistics.route_data_index",
    endpoint_name: str = "route-vs-endpoint"
):
    """
    RAG tool for route optimization data.

    Ryder Use Case: Real-time traffic, weather, delivery windows

    This tool retrieves route history and constraints for the
    Route Optimization Agent.
    """
    return create_vector_search_tool(
        index_name=index_name,
        endpoint_name=endpoint_name,
        description=(
            "Search route data including historical traffic patterns, delivery "
            "window constraints, and optimal paths. Use this to find route "
            "information for specific origins, destinations, or time windows."
        ),
        columns=["route_id", "origin", "destination", "avg_duration", "constraints"],
        num_results=5
    )


def create_vendor_data_rag_tool(
    index_name: str = "supply_chain.procurement.vendor_data_index",
    endpoint_name: str = "vendor-vs-endpoint"
):
    """
    RAG tool for vendor selection data.

    Ryder Use Case: Autonomous decision-making for vendor selection

    This tool retrieves vendor performance data for the
    Procurement Automation Agent.
    """
    return create_vector_search_tool(
        index_name=index_name,
        endpoint_name=endpoint_name,
        description=(
            "Search vendor data including performance scores, pricing history, "
            "reliability metrics, and compliance status. Use this to evaluate "
            "and compare vendors for procurement decisions."
        ),
        columns=["vendor_id", "vendor_name", "performance_score", "pricing_tier", "compliance_status"],
        num_results=10
    )


def create_risk_intelligence_rag_tool(
    index_name: str = "supply_chain.risk.disruption_data_index",
    endpoint_name: str = "risk-vs-endpoint"
):
    """
    RAG tool for supply chain risk intelligence.

    Ryder Use Case: Predictive analytics for supply chain disruptions

    This tool retrieves historical disruption data and risk factors
    for the Risk Management Agent.
    """
    return create_vector_search_tool(
        index_name=index_name,
        endpoint_name=endpoint_name,
        description=(
            "Search supply chain disruption history, risk factors, and mitigation "
            "strategies. Use this to identify potential risks, find similar past "
            "disruptions, and recommend mitigation approaches."
        ),
        columns=["incident_id", "risk_type", "impact_level", "mitigation_strategy", "lessons_learned"],
        num_results=8
    )
