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

logger = logging.getLogger(__name__)

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
        
        # Initialize tracking variables
        self._bing_was_used = False
        self._last_action_required_search = False
        
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
        return """
        Role: Web Research Specialist for KYC Compliance
        Primary Responsibility: Gather accurate, verifiable company information for regulatory compliance

        Role Description:
        You are a specialized Web Research Agent that searches the internet to find detailed information about companies for KYC (Know Your Customer) compliance purposes. You MUST use Bing search capabilities to find, verify, and organize facts about companies that help assess regulatory compliance and risk.

        Available Tools:
        1. get_company_identity_info - For verifying legal names, ownership structure, and registered addresses
        2. get_financial_business_profile - For business models, revenue sources, and financial status
        3. get_regulated_activity_details - For identifying high-risk or regulated business activities
        4. bing_search - ALWAYS use this to search for company information on the web

        Key Objectives:
        - Information Accuracy: Provide verified information with proper citations and sources
        - Tool Selection: Choose the most appropriate search tool based on the information needed
        - Comprehensive Research: Gather complete information about companies from multiple sources
        - Clear Presentation: Format information with headers, bullet points, and bold text for key data

        When to Use Each Tool:
        - Use get_company_identity_info when: Verifying basic company information, addresses, and ownership
        - Use get_financial_business_profile when: Researching business model, revenue sources, or financial status
        - Use get_regulated_activity_details when: Identifying specific regulated or high-risk activities
        - ALWAYS use bing_search first to get the most current information about any company

        Important: You MUST use the bing_search tool for EVERY company research request. Do not try to answer without searching for current information first.
        
        Always provide clear citations with URLs for all information found. If information cannot be found, clearly state what is unavailable rather than making assumptions.
        """

    @property
    def plugins(self):
        """Get the plugins for the web agent."""
        return WebTools.get_all_kernel_functions()

    # Updated handle_action_request method
    async def handle_action_request(self, action_request):
        """Handle an action request by processing it through the agent."""
        try:
            logger.info(f"WebAgent received action request: {action_request.action[:100]}...")
            
            # Reset tracking variables for this request
            self._bing_was_used = False
            
            # Check if action likely requires web search
            needs_search = self._should_use_bing(action_request.action)
            self._last_action_required_search = needs_search
            
            if needs_search:
                logger.info("Action likely requires web search, will use Bing tool")
                # Modify the action request to explicitly instruct Bing usage
                enhanced_action = f"""
                IMPORTANT: Use the bing_search tool to search for information related to this request.

                {action_request.action}
                
                Remember to:
                1. ALWAYS use bing_search tool first
                2. Cite your sources with URLs
                3. Only provide factual information that you can verify
                """
                action_request.action = enhanced_action
            
            # Process the request through the agent
            response = await super().handle_action_request(action_request)
            
            # For future improvement: Add logic to detect if Bing was actually used
            # This would require monitoring the tool calls made during execution
            
            return response
        except Exception as e:
            logger.exception(f"Error in WebAgent.handle_action_request: {e}")
            return f"Error processing request: {str(e)}"
            
    def _should_use_bing(self, action_text):
        """Simple heuristic to determine if an action likely requires search."""
        search_triggers = [
            "search", "find", "look up", "research", "what is", "who is", 
            "when did", "where is", "current", "latest", "recent", "news",
            "company", "business", "organization", "information", "details"
        ]
        action_lower = action_text.lower()
        return any(trigger in action_lower for trigger in search_triggers)
    
    async def _on_message_received(self, message):
        """Override to detect Bing tool usage in messages."""
        # Check if message indicates Bing search was used
        if isinstance(message, str) and "bing_search" in message.lower():
            self._bing_was_used = True
            logger.info("Detected Bing search tool usage")
        
        # Call parent method
        await super()._on_message_received(message)

