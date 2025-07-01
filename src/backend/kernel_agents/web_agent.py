import logging
from typing import List, Optional, Any
import re 
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

        IMPORTANT INSTRUCTIONS:
        1. You have access to web search capabilities through the bing_search tool
        2. When a function returns "EXECUTE SEARCH:" instructions, you MUST perform those searches
        3. DO NOT return the search instructions to the user - execute the searches and return formatted results
        4. Only execute searches for the SPECIFIC function that was called - do not call other functions
        5. Focus only on the information requested by the current function call

        Function Handling:
        - When get_company_identity_info() is called, search ONLY for identity information
        - When get_financial_business_profile() is called, search ONLY for financial information  
        - When get_regulated_activity_details() is called, search ONLY for regulatory information
        - Always format results according to the specific format requested by each function
        - Include proper source citations

        Search Process:
        1. Read the search instructions from the function result
        2. Execute targeted searches using the bing_search tool for ONLY the requested information
        3. Analyze and compile ONLY the search results relevant to the current function
        4. Format the final response according to the specified format
        5. Include proper source citations

        CRITICAL: Only search for and return information related to the specific function that was called.
        Do NOT execute multiple functions or mix information from different functions.
        """

    @property
    def plugins(self):
        """Get the plugins for the web agent."""
        return WebTools.get_all_kernel_functions()

    # Updated handle_action_request method
    async def handle_action_request(self, action_request):
        """Handle an action request by processing it through the agent."""
        try:
            logger.info(f"WebAgent received action request: {action_request.action}...")
            
            # Reset tracking variables for this request
            self._bing_was_used = False
            
            # Extract the specific function being called
            function_match = self.extract_function_name(action_request.action)
                        
            if function_match:
                enhanced_action = f"""
                IMPORTANT: Only focus on last assistant message and the function call.
                
                {action_request.action}
                
                Execute the search instructions returned by the function and provide formatted results.
                """
                action_request.action = enhanced_action
            
            logger.info(f"Processed action request for WebAgent with function: {function_match}")
            
            # Process the request through the agent
            response = await super().handle_action_request(action_request)
            
            return response
        except Exception as e:
            logger.exception(f"Error in WebAgent.handle_action_request: {e}")
            return f"Error processing request: {str(e)}"
    
    def extract_function_name(self, text):
        # Define the pattern to search for the function name
        pattern = r'<conversation_history \\\>.*?Function: (\w+)'
        # Search for the pattern in the text
        match = re.search(pattern, text)
        # If a match is found, return the function name
        if match:
            return match.group(1)
        else:
            return None
       
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

