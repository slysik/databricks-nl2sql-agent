# Databricks notebook source
# MAGIC %md
# MAGIC # Supply Chain Multi-Agent System - Databricks Demo
# MAGIC
# MAGIC This notebook demonstrates a multi-agent orchestration system for supply chain optimization,
# MAGIC equivalent to the Azure AI Foundry MACAE implementation but using Databricks Mosaic AI.
# MAGIC
# MAGIC **Ryder AI Use Cases Demonstrated:**
# MAGIC 1. Demand Forecasting - ML on historical data
# MAGIC 2. Route Optimization - Traffic, weather, delivery windows
# MAGIC 3. Autonomous Procurement - Vendor selection
# MAGIC 4. Risk Management - Disruption prediction
# MAGIC
# MAGIC **Architecture Pattern:**
# MAGIC ```
# MAGIC User Request → Supervisor Agent → Specialized Agents (parallel/sequential)
# MAGIC                                 ↓
# MAGIC                    [Demand] [Route] [Procurement] [Risk]
# MAGIC                                 ↓
# MAGIC                         Synthesized Response
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -U mlflow databricks-langchain langgraph==1.0.5 langchain-core

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import mlflow
from mlflow.models import ModelConfig

# Team configuration - equivalent to MACAE's JSON team configs
team_config = {
    "team_id": "team-supply-chain-1",
    "name": "Supply Chain Optimization Team",
    "model_serving_endpoint": "databricks-meta-llama-3-3-70b-instruct",
    "max_rounds": 10,
    "agents": [
        {
            "name": "DemandForecastAgent",
            "vector_search_index": "supply_chain.forecasting.historical_data_index",
            "vs_endpoint": "demand-vs-endpoint"
        },
        {
            "name": "RouteOptimizationAgent",
            "vector_search_index": "supply_chain.logistics.route_data_index",
            "vs_endpoint": "route-vs-endpoint"
        },
        {
            "name": "ProcurementAgent",
            "vector_search_index": "supply_chain.procurement.vendor_data_index",
            "vs_endpoint": "vendor-vs-endpoint"
        },
        {
            "name": "RiskManagementAgent",
            "vector_search_index": "supply_chain.risk.disruption_data_index",
            "vs_endpoint": "risk-vs-endpoint"
        }
    ]
}

model_config = mlflow.models.ModelConfig(development_config=team_config)
print(f"Team: {model_config.get('name')}")
print(f"Agents: {[a['name'] for a in model_config.get('agents')]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Agent Implementation

# COMMAND ----------

from typing import Annotated, List, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# Agent State - equivalent to ChatAgent message handling
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


class SupplyChainAgent:
    """
    Supply Chain Agent - Databricks equivalent to FoundryAgentTemplate.

    Key equivalences:
    - ChatDatabricks = AzureAIAgentClient
    - StateGraph = ChatAgent
    - tools list = mcp_tool + HostedCodeInterpreterTool
    """

    def __init__(self, name: str, instructions: str, tools: List = None):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []

        # Build LangGraph
        self._build_graph()

    def _build_graph(self):
        from langchain_databricks import ChatDatabricks

        # LLM - equivalent to model_deployment_name
        self.llm = ChatDatabricks(
            endpoint=model_config.get("model_serving_endpoint"),
            temperature=0.1
        )

        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

        def agent_node(state: AgentState):
            messages = [{"role": "system", "content": self.instructions}] + state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "end"

        builder = StateGraph(AgentState)
        builder.add_node("agent", agent_node)

        if self.tools:
            builder.add_node("tools", ToolNode(tools=self.tools))
            builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
            builder.add_edge("tools", "agent")
        else:
            builder.add_edge("agent", END)

        builder.add_edge(START, "agent")
        self._graph = builder.compile()

    def invoke(self, messages: list) -> dict:
        return self._graph.invoke({"messages": messages})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create RAG Tools (Vector Search)

# COMMAND ----------

# Mock RAG tools for demo - in production, use DatabricksVectorSearch

@tool
def search_demand_history(query: str) -> str:
    """Search historical demand data for forecasting analysis."""
    # In production: DatabricksVectorSearch retriever
    return f"""Historical Demand Data for: {query}

Product: Widget-A | Southwest Region
- Q1 2024: 15,000 units (10% YoY growth)
- Q4 2023: 18,500 units (holiday spike)
- Seasonality: Peak in Q4, trough in Q1
- Trend: Steady 8-12% annual growth

Product: Widget-B | Southwest Region
- Q1 2024: 8,200 units (flat YoY)
- Pattern: Stable demand, low volatility
"""

@tool
def search_route_data(query: str) -> str:
    """Search route and logistics data for optimization."""
    return f"""Route Analysis for: {query}

Phoenix Hub Routes:
- PHX → LA: Avg 6.5 hrs, Traffic peak 7-9 AM, 4-7 PM
- PHX → SD: Avg 5.2 hrs, I-8 construction delays expected
- PHX → Albuquerque: Avg 7 hrs, Weather risk in winter

Recommended: Depart before 5 AM for California routes
"""

@tool
def search_vendor_data(query: str) -> str:
    """Search vendor performance and pricing data."""
    return f"""Vendor Analysis for: {query}

Top Vendors (Packaging Materials):
1. PackCo Inc - Score: 92/100, Price: $$, Reliability: 98%
2. BoxMasters - Score: 87/100, Price: $, Reliability: 94%
3. Premium Pack - Score: 95/100, Price: $$$, Reliability: 99%

Recommendation: BoxMasters for cost efficiency, Premium Pack for critical shipments
"""

@tool
def search_risk_data(query: str) -> str:
    """Search supply chain risk and disruption history."""
    return f"""Risk Intelligence for: {query}

Summer Season Risks (Southwest):
- Heat-related delays: 15% increase in Q3
- Monsoon disruptions: July-August, affects PHX hub
- Driver availability: -8% during peak heat

Mitigation: Shift sensitive shipments to early AM, maintain buffer inventory
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Specialized Agents

# COMMAND ----------

# Agent instructions - equivalent to system_message in team JSON
DEMAND_INSTRUCTIONS = """You are the Demand Forecasting Agent.
Use the search_demand_history tool to analyze historical patterns.
Provide quantified forecasts with confidence levels."""

ROUTE_INSTRUCTIONS = """You are the Route Optimization Agent.
Use the search_route_data tool to find optimal paths.
Consider traffic, weather, and delivery windows."""

PROCUREMENT_INSTRUCTIONS = """You are the Procurement Agent.
Use the search_vendor_data tool to evaluate vendors.
Recommend based on price, quality, and reliability."""

RISK_INSTRUCTIONS = """You are the Risk Management Agent.
Use the search_risk_data tool to identify threats.
Provide actionable mitigation strategies."""

# Create agents - equivalent to MagenticAgentFactory.get_agents()
demand_agent = SupplyChainAgent("DemandForecastAgent", DEMAND_INSTRUCTIONS, [search_demand_history])
route_agent = SupplyChainAgent("RouteOptimizationAgent", ROUTE_INSTRUCTIONS, [search_route_data])
procurement_agent = SupplyChainAgent("ProcurementAgent", PROCUREMENT_INSTRUCTIONS, [search_vendor_data])
risk_agent = SupplyChainAgent("RiskManagementAgent", RISK_INSTRUCTIONS, [search_risk_data])

agents = {
    "DemandForecastAgent": demand_agent,
    "RouteOptimizationAgent": route_agent,
    "ProcurementAgent": procurement_agent,
    "RiskManagementAgent": risk_agent
}

print(f"Created {len(agents)} agents")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Build Supervisor Orchestration

# COMMAND ----------

from langgraph.checkpoint.memory import InMemorySaver

class OrchestratorState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    round_count: int
    agent_responses: dict
    final_result: str


class SupervisorOrchestrator:
    """
    Supervisor pattern - equivalent to OrchestrationManager with MagenticBuilder.

    Pattern:
        supervisor → agent1 → supervisor → agent2 → ... → synthesize
    """

    def __init__(self, agents: dict, max_rounds: int = 10):
        self.agents = agents
        self.max_rounds = max_rounds
        self.checkpointer = InMemorySaver()
        self._build()

    def _build(self):
        from langchain_databricks import ChatDatabricks

        supervisor_llm = ChatDatabricks(
            endpoint=model_config.get("model_serving_endpoint"),
            temperature=0.1
        )

        agent_names = list(self.agents.keys())
        supervisor_prompt = f"""You orchestrate these agents: {agent_names}
Decide which agent to call next based on the task.
Respond with ONLY the agent name or FINISH."""

        def supervisor_node(state):
            if state["round_count"] >= self.max_rounds:
                return {"next_agent": "FINISH"}

            msgs = [{"role": "system", "content": supervisor_prompt}] + state["messages"]
            response = supervisor_llm.invoke(msgs)
            next_agent = response.content.strip()

            if next_agent not in agent_names:
                next_agent = "FINISH"

            return {"next_agent": next_agent, "round_count": state["round_count"] + 1}

        def make_agent_node(name):
            def node(state):
                result = self.agents[name].invoke(state["messages"])
                content = result["messages"][-1].content
                return {
                    "messages": result["messages"],
                    "agent_responses": {**state["agent_responses"], name: content}
                }
            return node

        def route(state):
            return "synthesize" if state["next_agent"] == "FINISH" else state["next_agent"]

        def synthesize(state):
            summary = "\n\n".join([f"**{k}:**\n{v}" for k, v in state["agent_responses"].items()])
            return {"final_result": summary}

        # Build graph
        builder = StateGraph(OrchestratorState)
        builder.add_node("supervisor", supervisor_node)

        for name in agent_names:
            builder.add_node(name, make_agent_node(name))
            builder.add_edge(name, "supervisor")

        builder.add_node("synthesize", synthesize)
        builder.add_conditional_edges(
            "supervisor", route,
            {name: name for name in agent_names} | {"synthesize": "synthesize"}
        )
        builder.add_edge(START, "supervisor")
        builder.add_edge("synthesize", END)

        self.workflow = builder.compile(checkpointer=self.checkpointer)

    def run(self, task: str, thread_id: str = "demo"):
        config = {"configurable": {"thread_id": thread_id}}
        result = self.workflow.invoke({
            "messages": [{"role": "user", "content": task}],
            "next_agent": "",
            "round_count": 0,
            "agent_responses": {},
            "final_result": ""
        }, config=config)
        return result["final_result"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Run Orchestration Demo

# COMMAND ----------

# Create orchestrator - equivalent to OrchestrationManager.init_orchestration()
orchestrator = SupervisorOrchestrator(agents, max_rounds=10)

# Example task - equivalent to starting_tasks in team JSON
task = """
Analyze Southwest region operations for Q2 planning:
1. Forecast demand for top products
2. Optimize Phoenix hub routes
3. Evaluate packaging vendors
4. Assess summer season risks

Provide actionable recommendations.
"""

print("Starting multi-agent orchestration...\n")
result = orchestrator.run(task)
print("=" * 60)
print("FINAL ORCHESTRATED RESULT:")
print("=" * 60)
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Deploy to Model Serving

# COMMAND ----------

# Log model with MLflow - equivalent to Azure Container Apps deployment
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="supply_chain_agent",
        python_model=orchestrator,
        registered_model_name="supply_chain_multi_agent",
        pip_requirements=[
            "langgraph==1.0.5",
            "langchain-databricks",
            "mlflow>=2.18"
        ]
    )

print("Model logged. Deploy via Model Serving UI or API.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture Comparison Summary
# MAGIC
# MAGIC | MACAE (Azure AI Foundry) | Databricks Mosaic AI |
# MAGIC |--------------------------|----------------------|
# MAGIC | `FoundryAgentTemplate` | `SupplyChainAgent` (LangGraph) |
# MAGIC | `MagenticBuilder` | `StateGraph` supervisor |
# MAGIC | `HumanApprovalMagenticManager` | `interrupt()` function |
# MAGIC | Azure AI Search | Databricks Vector Search |
# MAGIC | `InMemoryCheckpointStorage` | `InMemorySaver` |
# MAGIC | Azure Container Apps | Model Serving Endpoint |
# MAGIC
# MAGIC **Key Insight for Interview:**
# MAGIC > "The patterns are identical - agent orchestration, RAG integration, human-in-the-loop.
# MAGIC > The difference is the underlying platform APIs. I've demonstrated I can translate
# MAGIC > between Azure AI Foundry and Databricks Mosaic AI frameworks."
