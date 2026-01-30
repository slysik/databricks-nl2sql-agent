"""
Human-in-the-Loop Approval for Databricks
Equivalent to MACAE's HumanApprovalMagenticManager

This implements approval checkpoints using LangGraph's interrupt() function.
"""

from typing import Optional, Dict, Any
from langgraph.types import interrupt, Command


def create_approval_checkpoint():
    """
    Create a tool/function that implements human approval checkpoint.

    Equivalent to HumanApprovalMagenticManager in MACAE:

    Azure AI Foundry (MACAE):
        manager = HumanApprovalMagenticManager(
            user_id=user_id,
            chat_client=chat_client,
        )
        # Manager intercepts execution for approval

    Databricks (This):
        approval_tool = create_approval_checkpoint()
        # Tool calls interrupt() to pause for human approval
    """
    from langchain_core.tools import tool

    @tool
    def request_human_approval(
        action: str,
        details: str,
        risk_level: str = "medium"
    ) -> str:
        """
        Request human approval before proceeding with an action.

        Use this when:
        - Making high-stakes decisions (vendor selection, large orders)
        - Taking irreversible actions
        - Recommendations exceed threshold values
        - Risk assessment indicates caution needed

        Args:
            action: The action requiring approval
            details: Detailed explanation of what will happen
            risk_level: low, medium, or high - affects urgency of review

        Returns:
            Approval status and any modifications from reviewer
        """
        # Create approval request payload
        approval_request = {
            "action": action,
            "details": details,
            "risk_level": risk_level,
            "message": f"Please review and approve: {action}",
            "requires_response": True
        }

        # Interrupt execution for human review
        # This is equivalent to:
        #   await connection_config.send_status_update_async({
        #       "type": WebsocketMessageType.APPROVAL_REQUEST,
        #       "data": approval_request
        #   })
        #   response = await orchestration_config.wait_for_approval(request_id)
        response = interrupt(approval_request)

        # Handle response
        if response is None:
            return "Approval request timed out. Action not taken."

        if response.get("approved"):
            modifications = response.get("modifications", {})
            if modifications:
                return f"APPROVED with modifications: {modifications}"
            return "APPROVED. Proceed with action."
        else:
            reason = response.get("reason", "No reason provided")
            return f"REJECTED: {reason}. Do not proceed."

    return request_human_approval


class ApprovalManager:
    """
    Manager class for handling approval workflows.

    Equivalent to HumanApprovalMagenticManager's state management:
    - Track pending approvals
    - Handle timeouts
    - Route responses to waiting operations
    """

    def __init__(self, timeout_seconds: int = 300):
        self.timeout_seconds = timeout_seconds
        self.pending_approvals: Dict[str, Any] = {}

    async def handle_interrupted_workflow(
        self,
        workflow,
        interrupt_data: dict,
        thread_id: str,
        approval_callback
    ):
        """
        Handle workflow interrupted for approval.

        This is called when the workflow yields an interrupt.
        The approval_callback should handle getting the actual approval
        (e.g., via WebSocket to frontend).

        Args:
            workflow: The LangGraph workflow instance
            interrupt_data: Data from the interrupt call
            thread_id: Thread identifier for workflow state
            approval_callback: Async function to get approval decision

        Returns:
            Final workflow result after approval handling
        """
        # Get approval decision (this is application-specific)
        # In MACAE, this goes through WebSocket
        approval_response = await approval_callback(interrupt_data)

        # Resume workflow with approval response
        config = {"configurable": {"thread_id": thread_id}}
        result = await workflow.ainvoke(
            Command(resume=approval_response),
            config=config
        )

        return result


# ===== Supply Chain Specific Approval Rules =====

def create_vendor_approval_tool():
    """
    Create approval tool specifically for vendor selection decisions.

    Ryder Use Case: Autonomous decision-making with governance
    """
    from langchain_core.tools import tool

    @tool
    def approve_vendor_selection(
        vendor_name: str,
        contract_value: float,
        risk_score: float,
        recommendation_summary: str
    ) -> str:
        """
        Request approval for vendor selection decision.

        Automatically escalates based on thresholds:
        - contract_value > $100,000: Requires senior approval
        - risk_score > 0.7: Requires risk review
        - New vendor: Requires procurement team review

        Args:
            vendor_name: Selected vendor name
            contract_value: Total contract value in USD
            risk_score: Risk assessment score (0-1)
            recommendation_summary: Brief explanation of selection

        Returns:
            Approval status
        """
        # Determine escalation level
        if contract_value > 100000:
            escalation = "senior_management"
        elif risk_score > 0.7:
            escalation = "risk_committee"
        else:
            escalation = "procurement_team"

        approval_request = {
            "type": "vendor_selection",
            "vendor_name": vendor_name,
            "contract_value": contract_value,
            "risk_score": risk_score,
            "recommendation": recommendation_summary,
            "escalation_level": escalation,
            "message": f"Approve vendor selection: {vendor_name} (${contract_value:,.2f})"
        }

        response = interrupt(approval_request)

        if response is None:
            return "Vendor selection approval timed out."

        if response.get("approved"):
            return f"Vendor '{vendor_name}' APPROVED for ${contract_value:,.2f} contract."
        else:
            return f"Vendor selection REJECTED: {response.get('reason', 'Not specified')}"

    return approve_vendor_selection


def create_route_change_approval_tool():
    """
    Create approval tool for significant route changes.

    Ryder Use Case: Route optimization with human oversight
    """
    from langchain_core.tools import tool

    @tool
    def approve_route_change(
        route_id: str,
        change_description: str,
        impact_assessment: str,
        estimated_savings_minutes: int
    ) -> str:
        """
        Request approval for route optimization changes.

        Required when:
        - Change affects multiple delivery windows
        - Route deviation exceeds 20%
        - Customer SLA impact possible

        Args:
            route_id: Identifier for the affected route
            change_description: What's being changed
            impact_assessment: Customer and operational impact
            estimated_savings_minutes: Expected time savings

        Returns:
            Approval status
        """
        approval_request = {
            "type": "route_change",
            "route_id": route_id,
            "change": change_description,
            "impact": impact_assessment,
            "savings": estimated_savings_minutes,
            "message": f"Approve route change: {change_description}"
        }

        response = interrupt(approval_request)

        if response is None:
            return "Route change approval timed out. Maintaining current route."

        if response.get("approved"):
            return f"Route change APPROVED. Implementing: {change_description}"
        else:
            alternate = response.get("alternate_action", "")
            if alternate:
                return f"Route change MODIFIED: {alternate}"
            return f"Route change REJECTED: {response.get('reason', 'Not specified')}"

    return approve_route_change
