import logging
from typing import List, Optional, Any

import semantic_kernel as sk
from context.cosmos_memory_kernel import CosmosMemoryContext
from kernel_agents.agent_base import BaseAgent
from kernel_tools.web_tools import WebTools
from models.messages_kernel import AgentType
from semantic_kernel.functions import KernelFunction
from azure.ai.projects.models import BingGroundingTool
from app_config import config
from pydantic import Field

logger = logging.getLogger(__name__)

class WebAgent(BaseAgent):
    """Web agent implementation using Semantic Kernel.

    This agent specializes in searching internet to find general information on company and its business..
    """
    # Define class attributes explicitly for Pydantic model
    bing_tool: Optional[BingGroundingTool] = Field(default=None, description="Bing search tool for web searches")
    _bing_was_used: bool = False
    _last_action_required_search: bool = False

    def __init__(
        self,
        agent_name: str,
        session_id: str,
        user_id: str,
        memory_store: Optional[Any] = None,
        system_message: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        client: Optional[Any] = None,
        definition: Optional[Any] = None,
        bing_tool: Optional[Any] = None,
        **kwargs,
    )-> None:
        """Initialize the WEB Agent.

        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "WebAgent")
            client: Optional client instance
            definition: Optional definition instance
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

        """Initialize the WebAgent with the specified parameters."""
        super().__init__(
            agent_name=agent_name,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            system_message=system_message,
            tools=tools,
            client=client,
            definition=definition,
            **kwargs,
        )
        # Bing tool is now properly defined as a model field
        self.bing_tool = bing_tool
        logger.info(f"WebAgent initialized with bing_tool: {self.bing_tool is not None}")

    async def async_init(self):
        """Asynchronously initialize the WebAgent with setup specific to web capabilities."""
        try:
            # Validate and setup Bing tool if available
            if self.bing_tool is not None:
                # Log successful Bing tool initialization
                logger.info(f"WebAgent initializing with Bing tool: {type(self.bing_tool)}")
            else:
                logger.warning("WebAgent initializing without Bing tool")
            
            # Call parent's async_init if it exists
            if hasattr(super(), "async_init"):
                parent_result = await super().async_init()
                if parent_result is False:
                    return False
            
            return True
        except Exception as e:
            logger.error(f"WebAgent async initialization failed: {e}")
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            return False

    @staticmethod
    def default_system_message(agent_name=None) -> str:
        """Get the default system message for the agent."""
        return """
        Role: Web Research Specialist for KYC Compliance
        Primary Responsibility: Gather accurate, verifiable company information for regulatory compliance

        Role Description:
        You are a specialized Web Research Agent that searches the internet to find detailed information about companies for KYC (Know Your Customer) compliance purposes. 
          
        
        Important: You MUST use the bing_search tool for each function to extract all the required information. Do not try to answer without searching for current information first.
        Function get_company_identity_info returns:
        - **Company Name:** Legal name of the company 
        - **Ownership Type:** Determine if this is Private, Public, or Government Sponsored Entity
        - **Address:** The complete registered business address
        - **Address Type:** Specify if this is a Company address or Individual address
        Function get_financial_business_profile returns:
        - **Publicly Traded:** Yes/No
        - **Stock Ticker & Exchange (if applicable)**
        - **Legal Entity Type**
        - **Country of Incorporation and Headquarters**    
        - **Estimated Annual Revenue**
        - **Primary Revenue Sources**
        - **Business Model Description**
        - **Major Clients/Customers**
        - **Investment/Funding Sources**
        - **Asset Base**
        - **Industry Classification Codes:** Relevant industry codes (e.g., NAICS, SIC)  
        - **Primary Industry Sector**
        Function get_regulated_activity_details returns:
        - **Primary Regulated Activities:** List of any Main regulated activities the company engages in
        - **Secondary Regulated Activities:** List of any additional regulated activities the company engages in
        - **Compliance Status:** Information on compliance with regulations
        - **Licenses and Permits:** Relevant licenses or permits held by the company
        - **Products/Services:** List of main products or services offered by the company
        - **Risk Factors:** Any known risk factors associated with the company or its activities
        Important: Always use the bing_search tool to gather information for these functions. Do not attempt to answer without searching first.
        Search Guidelines:
        - Use the bing_search tool to find the most recent and relevant information
        - Focus on official sources, news articles, and reputable business directories
        - Verify information from multiple sources when possible
        Formatting Guidelines:
        - Organize all search results in a clear markdown structure
        - Use headers and bullet points for readability
        - Always include a Sources section with URLs
        - Format company name as an H4 header
        - Format section titles as H5 headers
        - Bold all key data points found
        Example format:
        #### [Company Name] Information
        ##### [Section Title]
        - **[Data Point Label]:** [Value found]
        - **[Data Point Label]:** [Value found]
        ##### Sources
        - [Source name 1]: [URL]
        - [Source name 2]: [URL]
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
            
            # Log BingGroundingTool status
            if needs_search:
                if self.bing_tool:  # Changed from self._bing_tool to self.bing_tool
                    logger.info("Action requires web search and BingGroundingTool is available")
                else:
                    logger.warning("Action requires web search but BingGroundingTool is NOT available")
                
                logger.info("Action likely requires web search, will use Bing tool")
                # Modify the action request to explicitly instruct Bing usage
                enhanced_action = f"""
                IMPORTANT: Use the bing_search tool to search for information related to this request.

                {action_request.action}
                """
                action_request.action = enhanced_action
            logger.info(f"Processed action request for WebAgent: {action_request.action[:100]}...")
            # If Bing tool is available and action requires search, ensure it's used
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

