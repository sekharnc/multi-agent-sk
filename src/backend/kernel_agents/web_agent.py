import logging
from typing import List, Optional

import semantic_kernel as sk
from context.cosmos_memory_kernel import CosmosMemoryContext
from kernel_agents.agent_base import BaseAgent
from kernel_tools.generic_tools import GenericTools
from kernel_tools.web_tools import WebTools
from models.messages_kernel import AgentType
from semantic_kernel.functions import KernelFunction
from azure.ai.projects.models import BingGroundingTool
from app_config import config

class WebAgent(BaseAgent):
    """Web agent implementation using Semantic Kernel.

    This agent specializes in searching internet to find general information on company and its business..
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = AgentType.WEB.value,
        client=None,
        definition=None,
        bing_tool: Optional[BingGroundingTool] = None,
    ) -> None:
        """Initialize the Web Agent.

        Args:
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "WebAgent")
            client: Optional client instance
            definition: Optional definition instance
            bing_tool: Optional BingGroundingTool instance
"""
                # Load configuration if tools not provided
        if not tools:
            # Get tools directly from WebTools class
            tools_dict = WebTools.get_all_kernel_functions()
            tools = [KernelFunction.from_method(func) for func in tools_dict.values()]

        # Use system message from config if not explicitly provided
        if not system_message:
            system_message = self.default_system_message(agent_name)

        # Use agent name from config if available
        agent_name = AgentType.WEB.value
        
        # Store bing_tool for later use
        self._bing_tool = bing_tool
        
        # Call the parent initializer
        super().__init__(
            agent_name=agent_name,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=tools,
            system_message=system_message,
            client=client,
            definition=definition,
        )

    @staticmethod
    def default_system_message(agent_name=None) -> str:
        """Get the default system message for the agent."""
        return """You are an AI Agent that can help with searching the internet to answer general pre-KYC related questions about companies and their business. 

You have access to Bing search capabilities to find the information about:
- Company/Client Address Information
- Company/Client Business Information details
- Company/Client Business activity details
- Company/Client Business geography details
- Company/Client's AML and Risk Profile
- Company/Client's Environmental, Social or Governance (ESG) activities and details

Use the Bing search tool to find relevant, up-to-date information to answer user queries about companies and business topics. Always use citations to provide source information and links to source websites."""


    @property
    def plugins(self):
        """Get the plugins for the web agent."""
        return GenericTools.get_all_kernel_functions()

    async def handle_action_request(self, action_request_json: str) -> str:
        """Handle an action request from another agent or the system."""
        return await super().handle_action_request(action_request_json)
