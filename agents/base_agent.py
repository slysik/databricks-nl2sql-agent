"""
Base Agent Template for Databricks Mosaic AI
Equivalent to Azure AI Foundry's FoundryAgentTemplate

This demonstrates the same patterns used in MACAE but for Databricks.
"""

from typing import Annotated, Generator, List, Optional, TypedDict
from uuid import uuid4

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)


class AgentState(TypedDict):
    """State schema for LangGraph agent - equivalent to ChatAgent's message handling"""
    messages: Annotated[list, add_messages]


class DatabricksAgentTemplate(ResponsesAgent):
    """
    Base agent template for Databricks Mosaic AI Agent Framework.

    Equivalent to MACAE's FoundryAgentTemplate:
    - Supports RAG via Databricks Vector Search tools
    - Supports tool integration via Unity Catalog or custom tools
    - Follows MLflow ResponsesAgent interface for deployment

    Example - Compare to FoundryAgentTemplate:

    Azure AI Foundry (MACAE):
        agent = FoundryAgentTemplate(
            agent_name="ContractSummaryAgent",
            agent_instructions="You are the Summary Agent...",
            model_deployment_name="gpt-4.1-mini",
            search_config=SearchConfig(index_name="contract-summary-doc-index")
        )
        await agent.open()

    Databricks (This):
        agent = DatabricksAgentTemplate(
            agent_name="ContractSummaryAgent",
            instructions="You are the Summary Agent...",
            model_endpoint="databricks-meta-llama-3-3-70b-instruct",
            tools=[rag_tool]
        )
    """

    def __init__(
        self,
        agent_name: str,
        instructions: str,
        model_endpoint: str,
        tools: Optional[List] = None,
        temperature: float = 0.1,
    ):
        """
        Initialize the Databricks agent.

        Args:
            agent_name: Name of the agent (equivalent to agent_name in FoundryAgentTemplate)
            instructions: System instructions (equivalent to agent_instructions)
            model_endpoint: Databricks model serving endpoint name
            tools: List of LangChain tools (equivalent to MCP tools + RAG)
            temperature: LLM temperature setting
        """
        self.agent_name = agent_name
        self.instructions = instructions
        self.model_endpoint = model_endpoint
        self.tools = tools or []
        self.temperature = temperature
        self._graph = None

        # Initialize - equivalent to open() in FoundryAgentTemplate
        self._build_graph()

    def _build_graph(self):
        """
        Build LangGraph StateGraph.

        This is equivalent to _after_open() in FoundryAgentTemplate where
        ChatAgent is initialized with tools.
        """
        try:
            from langchain_databricks import ChatDatabricks
        except ImportError:
            # For local testing without Databricks
            from langchain_openai import ChatOpenAI as ChatDatabricks

        # Initialize LLM - equivalent to ChatAgent's chat_client
        self.llm = ChatDatabricks(
            endpoint=self.model_endpoint,
            temperature=self.temperature,
        )

        # Bind tools to LLM (equivalent to tool_choice="auto" in ChatAgent)
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

        def agent_node(state: AgentState):
            """
            Main agent processing node.
            Equivalent to ChatAgent.run_stream() processing.
            """
            # Prepend system instructions (equivalent to agent_instructions)
            system_msg = {"role": "system", "content": self.instructions}
            messages = [system_msg] + state["messages"]

            # Invoke LLM
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: AgentState) -> str:
            """
            Routing logic for tool calls.
            Equivalent to tool_choice="auto" behavior in ChatAgent.
            """
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"

        # Build the graph - equivalent to ChatAgent initialization
        builder = StateGraph(AgentState)

        # Add agent node
        builder.add_node("agent", agent_node)

        # Add tool node if tools are configured
        if self.tools:
            tool_node = ToolNode(tools=self.tools)
            builder.add_node("tools", tool_node)

            # Conditional routing based on tool calls
            builder.add_conditional_edges(
                "agent",
                should_continue,
                {"tools": "tools", "end": END}
            )

            # Loop back after tool execution
            builder.add_edge("tools", "agent")
        else:
            builder.add_edge("agent", END)

        # Entry point
        builder.add_edge(START, "agent")

        # Compile graph
        self._graph = builder.compile()

    def invoke(self, messages: list) -> dict:
        """
        Invoke agent with messages.
        Equivalent to FoundryAgentTemplate.invoke()
        """
        if not self._graph:
            raise RuntimeError("Agent not initialized; graph not built.")

        return self._graph.invoke({"messages": messages})

    async def ainvoke(self, messages: list) -> dict:
        """Async invoke"""
        if not self._graph:
            raise RuntimeError("Agent not initialized; graph not built.")

        return await self._graph.ainvoke({"messages": messages})

    def stream(self, messages: list):
        """
        Stream agent responses.
        Equivalent to FoundryAgentTemplate.invoke() with run_stream().
        """
        if not self._graph:
            raise RuntimeError("Agent not initialized; graph not built.")

        for chunk in self._graph.stream({"messages": messages}):
            yield chunk

    # ===== MLflow ResponsesAgent Interface =====
    # These methods enable deployment to Databricks Model Serving

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        MLflow ResponsesAgent predict method.
        Called when the deployed endpoint receives a request.
        """
        # Convert request messages to LangChain format
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.input
        ]

        # Invoke agent
        result = self.invoke(messages)

        # Format response
        last_message = result["messages"][-1]
        content = getattr(last_message, "content", str(last_message))

        return ResponsesAgentResponse(
            output=[{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": content}]
            }]
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Streaming predict method for MLflow ResponsesAgent.
        Equivalent to streaming WebSocket responses in MACAE.
        """
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.input
        ]

        item_id = str(uuid4())
        aggregated_content = ""

        for chunk in self.stream(messages):
            # Extract content from chunk
            if "agent" in chunk:
                message = chunk["agent"]["messages"][-1]
                content = getattr(message, "content", "")

                if content:
                    # Yield delta event
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.delta",
                        item_id=item_id,
                        delta={"type": "text_delta", "text": content}
                    )
                    aggregated_content += content

        # Yield completion event
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item={
                "type": "message",
                "role": "assistant",
                "id": item_id,
                "content": [{"type": "text", "text": aggregated_content}]
            }
        )


# ===== Factory Function =====
# Equivalent to create_foundry_agent() factory

def create_databricks_agent(
    agent_name: str,
    instructions: str,
    model_endpoint: str,
    tools: Optional[List] = None,
) -> DatabricksAgentTemplate:
    """
    Factory for creating Databricks agents.
    Equivalent to create_foundry_agent() in MACAE.

    Usage:
        agent = create_databricks_agent(
            agent_name="ContractSummaryAgent",
            instructions="You are the Summary Agent...",
            model_endpoint="databricks-meta-llama-3-3-70b-instruct",
            tools=[rag_tool]
        )
    """
    return DatabricksAgentTemplate(
        agent_name=agent_name,
        instructions=instructions,
        model_endpoint=model_endpoint,
        tools=tools,
    )
