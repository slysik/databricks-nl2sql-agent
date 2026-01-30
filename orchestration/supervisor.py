"""
Multi-Agent Supervisor Orchestration for Databricks
Equivalent to MACAE's OrchestrationManager using MagenticBuilder

This implements the supervisor pattern for coordinating multiple
specialized agents, mirroring the Azure AI Foundry orchestration.
"""

from typing import Annotated, Dict, List, Literal, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


class OrchestratorState(TypedDict):
    """
    State for the supervisor orchestration graph.

    Equivalent to the state managed by MagenticBuilder:
    - messages: Conversation history (equivalent to ChatHistory)
    - next_agent: Routing decision (equivalent to MagenticOrchestratorMessageEvent)
    - round_count: Iteration tracking (equivalent to max_round_count)
    - agent_responses: Collected responses (equivalent to MagenticAgentMessageEvent)
    - final_result: Synthesized output (equivalent to MagenticFinalResultEvent)
    """
    messages: Annotated[list, add_messages]
    next_agent: str
    round_count: int
    agent_responses: Dict[str, str]
    final_result: str


class DatabricksOrchestrationManager:
    """
    Multi-agent orchestration manager using LangGraph supervisor pattern.

    Equivalent to MACAE's OrchestrationManager:

    Azure AI Foundry (MACAE):
        workflow = await OrchestrationManager.init_orchestration(
            agents=agents,
            team_config=team_config,
            memory_store=memory_store,
            user_id=user_id
        )
        async for event in workflow.run_stream(task_text):
            handle_event(event)

    Databricks (This):
        orchestrator = DatabricksOrchestrationManager(
            agents={"SummaryAgent": summary_agent, ...},
            supervisor_endpoint="databricks-meta-llama-3-3-70b-instruct",
            max_rounds=10
        )
        async for event in orchestrator.run_stream(task_text, thread_id):
            handle_event(event)
    """

    def __init__(
        self,
        agents: Dict[str, any],  # {"agent_name": agent_instance}
        supervisor_endpoint: str,
        max_rounds: int = 10,
        user_id: Optional[str] = None,
    ):
        """
        Initialize the orchestration manager.

        Args:
            agents: Dictionary mapping agent names to agent instances
            supervisor_endpoint: Databricks model endpoint for supervisor LLM
            max_rounds: Maximum orchestration rounds (equivalent to max_round_count)
            user_id: User identifier for state isolation
        """
        self.agents = agents
        self.supervisor_endpoint = supervisor_endpoint
        self.max_rounds = max_rounds
        self.user_id = user_id

        # Checkpointer - equivalent to InMemoryCheckpointStorage
        self.checkpointer = InMemorySaver()

        # Build the supervisor graph
        self._build_supervisor_graph()

    def _build_supervisor_graph(self):
        """
        Build the supervisor orchestration graph.

        Equivalent to MagenticBuilder.build():
            builder = (
                MagenticBuilder()
                .participants(**participants)
                .with_standard_manager(manager=manager, max_round_count=max_rounds)
                .with_checkpointing(storage)
            )
            workflow = builder.build()
        """
        try:
            from langchain_databricks import ChatDatabricks
        except ImportError:
            from langchain_openai import ChatOpenAI as ChatDatabricks

        # Supervisor LLM
        supervisor_llm = ChatDatabricks(
            endpoint=self.supervisor_endpoint,
            temperature=0.1
        )

        agent_names = list(self.agents.keys())
        agent_descriptions = self._get_agent_descriptions()

        # Supervisor system prompt
        supervisor_prompt = f"""You are a supervisor orchestrating a team of specialized agents.

Available agents and their capabilities:
{agent_descriptions}

Your job is to:
1. Analyze the user's request
2. Decide which agent should handle the current step
3. Continue until the task is complete, then respond with FINISH

Rules:
- Call agents in logical order based on the task
- Each agent should contribute something new
- Don't call the same agent twice unless necessary
- Synthesize final results when all required work is done

Respond with ONLY the agent name to call next, or FINISH if done.
Valid responses: {', '.join(agent_names)}, FINISH"""

        def supervisor_node(state: OrchestratorState) -> dict:
            """
            Supervisor decision node.
            Equivalent to HumanApprovalMagenticManager orchestration logic.
            """
            # Check round limit (equivalent to max_round_count)
            if state["round_count"] >= self.max_rounds:
                return {"next_agent": "FINISH"}

            # Build supervisor context
            context_msgs = [
                {"role": "system", "content": supervisor_prompt}
            ] + state["messages"]

            # If we have agent responses, add summary
            if state["agent_responses"]:
                summary = "Previous agent contributions:\n"
                for agent, response in state["agent_responses"].items():
                    summary += f"- {agent}: {response[:200]}...\n"
                context_msgs.append({
                    "role": "assistant",
                    "content": summary
                })

            # Get supervisor decision
            response = supervisor_llm.invoke(context_msgs)
            next_agent = response.content.strip()

            # Validate response
            if next_agent not in agent_names and next_agent != "FINISH":
                # Default to FINISH if invalid response
                next_agent = "FINISH"

            return {
                "next_agent": next_agent,
                "round_count": state["round_count"] + 1
            }

        def create_agent_node(agent_name: str):
            """
            Create node that invokes a specific agent.
            Equivalent to how MagenticBuilder calls participants.
            """
            def agent_node(state: OrchestratorState) -> dict:
                agent = self.agents[agent_name]

                # Invoke agent with current messages
                # Agents expect {"messages": [...]} format
                result = agent.invoke(state["messages"])

                # Extract response
                last_message = result["messages"][-1]
                content = getattr(last_message, "content", str(last_message))

                # Update state
                return {
                    "messages": result["messages"],
                    "agent_responses": {
                        **state["agent_responses"],
                        agent_name: content
                    }
                }
            return agent_node

        def route_to_agent(state: OrchestratorState) -> str:
            """
            Route based on supervisor decision.
            Equivalent to MagenticBuilder's workflow routing.
            """
            if state["next_agent"] == "FINISH":
                return "synthesize"
            return state["next_agent"]

        def synthesize_node(state: OrchestratorState) -> dict:
            """
            Synthesize final result from all agent contributions.
            Equivalent to MagenticFinalResultEvent handling.
            """
            if not state["agent_responses"]:
                return {"final_result": "No agents were called to process this request."}

            # Build synthesis prompt
            synthesis_prompt = """Synthesize the following agent contributions into a comprehensive final response:

"""
            for agent, response in state["agent_responses"].items():
                synthesis_prompt += f"### {agent} Analysis:\n{response}\n\n"

            synthesis_prompt += "\nProvide a unified, well-structured final response."

            # Use supervisor LLM for synthesis
            result = supervisor_llm.invoke([
                {"role": "system", "content": "You synthesize multiple agent outputs into coherent responses."},
                {"role": "user", "content": synthesis_prompt}
            ])

            return {"final_result": result.content}

        # Build the graph
        builder = StateGraph(OrchestratorState)

        # Add supervisor node
        builder.add_node("supervisor", supervisor_node)

        # Add agent nodes
        for name in agent_names:
            builder.add_node(name, create_agent_node(name))
            # Each agent returns to supervisor
            builder.add_edge(name, "supervisor")

        # Add synthesis node
        builder.add_node("synthesize", synthesize_node)

        # Add conditional routing from supervisor
        routing_map = {name: name for name in agent_names}
        routing_map["synthesize"] = "synthesize"
        builder.add_conditional_edges("supervisor", route_to_agent, routing_map)

        # Entry and exit points
        builder.add_edge(START, "supervisor")
        builder.add_edge("synthesize", END)

        # Compile with checkpointing
        self.workflow = builder.compile(checkpointer=self.checkpointer)

    def _get_agent_descriptions(self) -> str:
        """Get formatted agent descriptions for supervisor prompt"""
        descriptions = []
        for name, agent in self.agents.items():
            desc = getattr(agent, "instructions", "No description available")
            # Truncate for prompt efficiency
            if len(desc) > 200:
                desc = desc[:200] + "..."
            descriptions.append(f"- {name}: {desc}")
        return "\n".join(descriptions)

    async def run_stream(self, task: str, thread_id: str):
        """
        Execute orchestration with streaming.

        Equivalent to OrchestrationManager.run_orchestration():
            async for event in workflow.run_stream(task_text):
                if isinstance(event, MagenticAgentDeltaEvent):
                    await streaming_agent_response_callback(...)
                elif isinstance(event, MagenticFinalResultEvent):
                    handle_final_result(event)

        Yields event dictionaries with type information.
        """
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "messages": [{"role": "user", "content": task}],
            "next_agent": "",
            "round_count": 0,
            "agent_responses": {},
            "final_result": ""
        }

        async for event in self.workflow.astream(initial_state, config=config):
            # Yield events with type information for callback handling
            for node_name, node_output in event.items():
                yield {
                    "type": "node_output",
                    "node": node_name,
                    "data": node_output
                }

    def run(self, task: str, thread_id: str) -> str:
        """
        Synchronous execution.
        Returns final synthesized result.
        """
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "messages": [{"role": "user", "content": task}],
            "next_agent": "",
            "round_count": 0,
            "agent_responses": {},
            "final_result": ""
        }

        result = self.workflow.invoke(initial_state, config=config)
        return result.get("final_result", "")


# ===== Factory Function =====

def create_orchestration(
    agents: Dict[str, any],
    supervisor_endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
    max_rounds: int = 10,
    user_id: Optional[str] = None,
) -> DatabricksOrchestrationManager:
    """
    Factory for creating orchestration manager.

    Equivalent to OrchestrationManager.init_orchestration():

        workflow = await OrchestrationManager.init_orchestration(
            agents=agents,
            team_config=team_config,
            memory_store=memory_store,
            user_id=user_id
        )

    Usage:
        orchestrator = create_orchestration(
            agents={
                "SummaryAgent": summary_agent,
                "RiskAgent": risk_agent,
                "ComplianceAgent": compliance_agent
            },
            max_rounds=10,
            user_id="user-123"
        )
    """
    return DatabricksOrchestrationManager(
        agents=agents,
        supervisor_endpoint=supervisor_endpoint,
        max_rounds=max_rounds,
        user_id=user_id
    )
