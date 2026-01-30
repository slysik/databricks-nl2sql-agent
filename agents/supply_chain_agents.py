"""
Supply Chain Agents for Databricks - Ryder Use Cases

These agents map to Ryder's AI initiatives:
1. Demand Forecasting - ML on historical data for inventory optimization
2. Route Optimization - Real-time traffic, weather, delivery windows
3. Autonomous Decision-Making - Vendor selection, procurement automation
4. Risk Management - Predictive analytics for supply chain disruptions

Compare to MACAE's Contract Compliance Team structure.
"""

from typing import List, Optional
from databricks_prototype.agents.base_agent import DatabricksAgentTemplate, create_databricks_agent
from databricks_prototype.tools.rag_tools import (
    create_demand_forecast_rag_tool,
    create_route_data_rag_tool,
    create_vendor_data_rag_tool,
    create_risk_intelligence_rag_tool,
)


# ===== Agent Definitions =====
# Structure mirrors MACAE's contract_compliance_team.json

SUPPLY_CHAIN_TEAM_CONFIG = {
    "team_id": "team-supply-chain-1",
    "name": "Supply Chain Optimization Team",
    "model_serving_endpoint": "databricks-meta-llama-3-3-70b-instruct",
    "description": "A multi-agent team that optimizes supply chain operations through demand forecasting, route optimization, vendor selection, and risk assessment.",
    "agents": [
        {
            "name": "DemandForecastAgent",
            "type": "forecasting",
            "system_message": """You are the Demand Forecasting Agent for supply chain optimization.

Your task is to analyze historical demand patterns and generate accurate forecasts.

Responsibilities:
1. Retrieve and analyze historical demand data using the search tool
2. Identify seasonal patterns, trends, and anomalies
3. Generate demand forecasts for specified products/regions/timeframes
4. Quantify forecast confidence levels
5. Recommend inventory adjustments based on predictions

Output Structure:
- Demand Forecast Summary
- Key Patterns Identified
- Confidence Level (High/Medium/Low)
- Recommended Actions
- Risk Factors

Always ground your analysis in the retrieved historical data. Do not hallucinate numbers.""",
            "use_rag": True,
            "rag_index": "supply_chain.forecasting.historical_data_index",
            "rag_endpoint": "demand-vs-endpoint"
        },
        {
            "name": "RouteOptimizationAgent",
            "type": "routing",
            "system_message": """You are the Route Optimization Agent for logistics operations.

Your task is to optimize delivery routes considering multiple constraints.

Responsibilities:
1. Analyze route history and traffic patterns using the search tool
2. Factor in delivery windows, priorities, and capacity constraints
3. Identify optimal paths considering current conditions
4. Calculate estimated arrival times with confidence ranges
5. Suggest route alternatives for contingencies

Input Factors to Consider:
- Traffic patterns (historical + real-time indicators)
- Weather conditions
- Delivery window requirements
- Vehicle capacity and type
- Priority levels
- Driver hours of service

Output Structure:
- Optimized Route Recommendation
- Estimated Duration & ETA
- Key Constraints Applied
- Alternative Routes
- Risk Factors (weather, traffic)

Be precise with time estimates and always explain your routing logic.""",
            "use_rag": True,
            "rag_index": "supply_chain.logistics.route_data_index",
            "rag_endpoint": "route-vs-endpoint"
        },
        {
            "name": "ProcurementAgent",
            "type": "procurement",
            "system_message": """You are the Procurement Automation Agent for vendor selection and purchasing decisions.

Your task is to provide data-driven vendor recommendations and procurement analysis.

Responsibilities:
1. Retrieve vendor performance data using the search tool
2. Compare vendors on price, quality, reliability, and compliance
3. Assess total cost of ownership (not just unit price)
4. Identify potential supply risks with specific vendors
5. Recommend optimal vendor selection with justification

Evaluation Criteria:
- Performance Score (delivery accuracy, quality metrics)
- Pricing competitiveness
- Reliability history
- Compliance status (certifications, regulatory)
- Risk exposure (single source, geographic)

Output Structure:
- Vendor Comparison Matrix
- Recommended Vendor(s)
- Decision Rationale
- Risk Assessment
- Contract Considerations

Always justify decisions with data from the vendor database.""",
            "use_rag": True,
            "rag_index": "supply_chain.procurement.vendor_data_index",
            "rag_endpoint": "vendor-vs-endpoint"
        },
        {
            "name": "RiskManagementAgent",
            "type": "risk",
            "system_message": """You are the Risk Management Agent for supply chain disruption analysis.

Your task is to identify, assess, and recommend mitigations for supply chain risks.

Responsibilities:
1. Search historical disruption data using the search tool
2. Identify current risk factors and potential disruptions
3. Assess impact severity and probability
4. Recommend proactive mitigation strategies
5. Learn from past incidents to prevent recurrence

Risk Categories to Monitor:
- Supplier risks (single source, financial stability)
- Transportation risks (weather, infrastructure, capacity)
- Demand risks (volatility, forecast accuracy)
- Regulatory risks (compliance, tariffs, sanctions)
- Operational risks (facility, labor, technology)

Output Structure:
- Risk Assessment Summary
- Identified Risks (High/Medium/Low severity)
- Historical Precedents (similar past incidents)
- Mitigation Recommendations
- Monitoring Triggers

Provide actionable intelligence, not generic warnings.""",
            "use_rag": True,
            "rag_index": "supply_chain.risk.disruption_data_index",
            "rag_endpoint": "risk-vs-endpoint"
        }
    ]
}


def create_demand_forecast_agent(
    model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
) -> DatabricksAgentTemplate:
    """
    Create the Demand Forecasting Agent.

    Ryder Use Case: ML on historical data for inventory optimization
    """
    config = SUPPLY_CHAIN_TEAM_CONFIG["agents"][0]

    # Create RAG tool
    rag_tool = create_demand_forecast_rag_tool(
        index_name=config["rag_index"],
        endpoint_name=config["rag_endpoint"]
    )

    return create_databricks_agent(
        agent_name=config["name"],
        instructions=config["system_message"],
        model_endpoint=model_endpoint,
        tools=[rag_tool]
    )


def create_route_optimization_agent(
    model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
) -> DatabricksAgentTemplate:
    """
    Create the Route Optimization Agent.

    Ryder Use Case: Real-time traffic, weather, delivery windows
    """
    config = SUPPLY_CHAIN_TEAM_CONFIG["agents"][1]

    rag_tool = create_route_data_rag_tool(
        index_name=config["rag_index"],
        endpoint_name=config["rag_endpoint"]
    )

    return create_databricks_agent(
        agent_name=config["name"],
        instructions=config["system_message"],
        model_endpoint=model_endpoint,
        tools=[rag_tool]
    )


def create_procurement_agent(
    model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
) -> DatabricksAgentTemplate:
    """
    Create the Procurement Automation Agent.

    Ryder Use Case: Autonomous decision-making for vendor selection
    """
    config = SUPPLY_CHAIN_TEAM_CONFIG["agents"][2]

    rag_tool = create_vendor_data_rag_tool(
        index_name=config["rag_index"],
        endpoint_name=config["rag_endpoint"]
    )

    return create_databricks_agent(
        agent_name=config["name"],
        instructions=config["system_message"],
        model_endpoint=model_endpoint,
        tools=[rag_tool]
    )


def create_risk_management_agent(
    model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
) -> DatabricksAgentTemplate:
    """
    Create the Risk Management Agent.

    Ryder Use Case: Predictive analytics for supply chain disruptions
    """
    config = SUPPLY_CHAIN_TEAM_CONFIG["agents"][3]

    rag_tool = create_risk_intelligence_rag_tool(
        index_name=config["rag_index"],
        endpoint_name=config["rag_endpoint"]
    )

    return create_databricks_agent(
        agent_name=config["name"],
        instructions=config["system_message"],
        model_endpoint=model_endpoint,
        tools=[rag_tool]
    )


def create_supply_chain_team(
    model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
) -> dict:
    """
    Create the full Supply Chain Optimization Team.

    Equivalent to MACAE's MagenticAgentFactory.get_agents():

        agents = await factory.get_agents(
            user_id=user_id,
            team_config_input=team_config,
            memory_store=team_service.memory_context
        )

    Usage:
        agents = create_supply_chain_team()
        orchestrator = create_orchestration(agents=agents)
    """
    return {
        "DemandForecastAgent": create_demand_forecast_agent(model_endpoint),
        "RouteOptimizationAgent": create_route_optimization_agent(model_endpoint),
        "ProcurementAgent": create_procurement_agent(model_endpoint),
        "RiskManagementAgent": create_risk_management_agent(model_endpoint),
    }


# ===== Example Usage =====

def demo_supply_chain_orchestration():
    """
    Demonstration of supply chain multi-agent orchestration.

    This shows the same pattern as MACAE but for Ryder's use cases.
    """
    from databricks_prototype.orchestration.supervisor import create_orchestration

    # Create agent team
    agents = create_supply_chain_team()

    # Create orchestrator
    orchestrator = create_orchestration(
        agents=agents,
        max_rounds=10,
        user_id="demo-user"
    )

    # Example task (like MACAE's starting_tasks)
    task = """
    Analyze our Q1 supply chain operations and provide recommendations:

    1. Forecast demand for our top 10 products in the Southwest region for Q2
    2. Optimize our primary distribution routes from Phoenix hub
    3. Evaluate our current vendors for packaging materials
    4. Assess risks for the upcoming summer season

    Provide a comprehensive report with actionable recommendations.
    """

    # Run orchestration
    result = orchestrator.run(task, thread_id="demo-thread")
    print(result)

    return result
