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
        bing_tool: Optional[BingGroundingTool] = None,  # Add this parameter
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
        
        # Add Bing tool to tools list if provided
        if bing_tool and hasattr(bing_tool, 'definitions'):
            if isinstance(tools, list):
                tools.extend(bing_tool.definitions)
            else:
                tools = list(bing_tool.definitions)
        
        # Ensure definition is not None - create a basic one if needed
        if definition is None:
            # Create a minimal definition object or use agent_name
            # This depends on what type of definition object is expected
            # You may need to import the appropriate definition class
            definition = type('Definition', (), {'name': agent_name})()
        
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
        """Get the default system message for the agent.
        Args:
            agent_name: The name of the agent (optional)
        Returns:
            The default system message for the agent
        """
        return "You are an AI Agent that can help with searching internet to answer general questions about company and its business. You have BingGroundingTools available to you for this purpose."

    @property
    def plugins(self):
        """Get the plugins for the web agent."""
        return GenericTools.get_all_kernel_functions()

    # Explicitly inherit handle_action_request from the parent class
    async def handle_action_request(self, action_request_json: str) -> str:
        """Handle an action request from another agent or the system.

        This method is inherited from BaseAgent but explicitly included here for clarity.

        Args:
            action_request_json: The action request as a JSON string

        Returns:
            A JSON string containing the action response
        """
        return await super().handle_action_request(action_request_json)
