import logging
from typing import List, Optional, Any
from azure.ai.projects.models import AzureAISearchTool  # Import AzureAISearchTool
from pydantic import BaseModel, Field
import traceback
import semantic_kernel as sk
from context.cosmos_memory_kernel import CosmosMemoryContext
from kernel_agents.agent_base import BaseAgent
from kernel_tools.enterprise_tools import EnterpriseTools
from models.messages_kernel import AgentType
from semantic_kernel.functions import KernelFunction
import json
from app_config import config


logger = logging.getLogger(__name__)

class EnterpriseAgent(BaseAgent):
    """Enterprise agent implementation using Semantic Kernel."""

    # Define class attributes explicitly for Pydantic model
    search_tool: Optional[AzureAISearchTool] = Field(default=None, description="AzureAISearch tool for document search")
    _search_was_used: bool = False
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
        search_tool: Optional[Any] = None,
        **kwargs,
    )-> None:
        """Initialize the Enterprise Agent."""
        # Load configuration if tools not provided
        if not tools:
            # Get tools directly from EnterpriseTools class
            tools_dict = EnterpriseTools.get_all_kernel_functions()
            tools = [KernelFunction.from_method(func) for func in tools_dict.values()]

        # Use system message from config if not explicitly provided
        if not system_message:
            system_message = self.default_system_message(agent_name)

        # Use agent name from config if available
        agent_name = AgentType.ENTERPRISE.value
        tools = tools or []
        if search_tool:
            tools.append(search_tool)
            
        # Call parent initializer
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
        # Store search tool
        self.search_tool = search_tool
        logger.info(f"EnterpriseAgent initialized with search_tool: {self.search_tool is not None}")
    async def async_init(self):
        """Initialize enterprise agent with search capabilities."""
        
        # Get the search tool first
        search_tool = await config.get_azure_ai_search_tool()
        if search_tool: 
            # Store the search tool for direct access
            self.search_tool = search_tool
            
            # Add search tool to the tools list
            if self._tools:
                self._tools.append(search_tool)
            else:
                self._tools = [search_tool]
            logging.info(f"Added Azure AI Search tool to {self._agent_name}")
        else:
            logging.warning(f"Failed to get Azure AI Search tool for {self._agent_name}")
        
        # Call the base class init
        result = await super().async_init()
        
        # Verify tool registration
        if hasattr(self, '_agent') and self._agent:
            # Check if the agent has the search tool registered
            try:
                agent_def = await self.client.agents.get_agent(self.definition.id)
                tools = getattr(agent_def, 'tools', [])
                search_found = any(
                    isinstance(tool, dict) and tool.get('name') == 'azure_ai_search' 
                    for tool in tools
                )
                if search_found:
                    logging.info("Verified AzureAISearchTool is registered with agent")
                else:
                    logging.error("AzureAISearchTool not found in agent tools after initialization")
            except Exception as e:
                logging.error(f"Error verifying tool registration: {e}")
        
        return result
    
    @staticmethod
    def default_system_message(agent_name=None) -> str:
        """Get the default system message for the agent."""
        return """
        Role: Enterprise Information Specialist for KYC Compliance
        Primary Responsibility: Access and analyze internal company documents for regulatory compliance
        
        You are an Enterprise Agent that searches internal documents to provide accurate information for compliance purposes.
        
        IMPORTANT: You MUST use the AzureAISearch tool for all information requests. 
        
        When using the get_internal_risk_details function:
        - You must provide the country name to search for
        - The function will search internal documents for sanctions and risk information
        - Always return properly formatted results with citations to the source documents
        
        Guidelines:
        - Always use AzureAISearch tool when asked about country risk, sanctions, or compliance information
        - Cite document sources in your responses
        - Format information clearly with headers and bullet points
        - If information is not found, clearly state so and suggest alternative search terms
        
        Example response format:
        #### [Country] Risk and Sanctions Information
        ##### Risk Category
        - **Risk Level:** [Level found]
        - **Sanctions Status:** [Status found]
                
        ##### Sources
        - [Document name 1]: Internal document ID [xxx]
        - [Document name 2]: Internal document ID [xxx]
        """

    @property
    def plugins(self):
        """Get the plugins for the enterprise agent."""
        return EnterpriseTools.get_all_kernel_functions()

    # Explicitly inherit handle_action_request from the parent class
    async def handle_action_request(self, action_request):
        """Handle an action request by processing it through the agent."""
        try:
            logger.info(f"EnterpriseAgent received action request")
            
            # Check if action likely requires search
            needs_search = self._should_use_search(action_request.action)
            self._last_action_required_search = needs_search
            
            # Log search tool status
            if needs_search:
                logger.info(f"Action likely requires search, search_tool available: {self.search_tool is not None}")
                
                # Make the enhancement much stronger
                original_action = action_request.action
                enhanced_action = f"""
                === CRITICAL INSTRUCTION ===
                You MUST use the AzureAISearchTool to search for information in our internal database.
                
                STEP 1: ALWAYS search using AzureAISearchTool with a query matching this request:
                * For country risk details: Use 'country sanctions risk category [country name]'
                * For general sanctions: Use 'sanctions [relevant terms]'
                
                STEP 2: Show the search was performed by starting your response with "I searched our internal database..."
                
                STEP 3: Format your final answer based on search results only.
                
                Original request: {original_action}
                """
                action_request.action = enhanced_action
                logger.info("Enhanced action with CRITICAL search requirements")

            # Process the request through the agent
            response = await super().handle_action_request(action_request)
            
            # Check response for evidence of search usage
            if needs_search:
                response_obj = json.loads(response)
                if "result" in response_obj:
                    result_text = response_obj["result"].lower()
                    search_indicators = ["search", "internal database", "azure ai search", "found in our database"]
                    search_used = any(indicator in result_text for indicator in search_indicators)
                    logger.info(f"Evidence that search was used: {search_used}")
                    self._search_was_used = search_used
            
            return response
            
        except Exception as e:
            logger.error(f"Error in EnterpriseAgent.handle_action_request: {e}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
            raise
    def _should_use_search(self, action_text):
        """Simple heuristic to determine if an action likely requires search."""
        search_triggers = [
            "search", "find", "look up", "research", "information", "details",
            "risk", "category", "sanction", "country", "internal", "document",
            "get internal risk details"
        ]
        action_lower = action_text.lower()
        return any(trigger in action_lower for trigger in search_triggers)
    
    def set_search_tool(self, search_tool):
        """Set the AzureAISearchTool after initialization"""
        self.search_tool = search_tool
        # Register the search tool with the kernel
        if search_tool and hasattr(self, "kernel"):
            try:
                # Check which API style the kernel.plugins uses
                if hasattr(self.kernel.plugins, "add_plugin"):
                    # New style API
                    self.kernel.plugins.add_plugin("AzureAISearch", search_tool)
                else:
                    # Old style API - plugins might be a dict
                    self.kernel.plugins["AzureAISearch"] = search_tool
                
                logger.info("AzureAISearch plugin successfully registered with kernel")
            except Exception as e:
                logger.error(f"Error registering AzureAISearch plugin with kernel: {e}")
                # Log detailed information for debugging
                logger.error(f"Search tool type: {type(search_tool)}")
                logger.error(f"Kernel plugins type: {type(self.kernel.plugins)}")
                
                # Check if search tool has expected methods
                if hasattr(search_tool, 'search'):
                    logger.info("Search tool has 'search' method")
                else:
                    logger.error("Search tool doesn't have 'search' method")
    
    def verify_search_plugin(self):
        """Verify that the search plugin is properly registered and configured."""
        logger.info("Verifying search plugin configuration...")
        
        # Check if search_tool exists
        if not hasattr(self, "search_tool") or self.search_tool is None:
            logger.error("No search_tool attribute found on EnterpriseAgent")
            return False
        
        # Check if search_tool has search method
        if not hasattr(self.search_tool, "search"):
            logger.error("search_tool doesn't have a 'search' method")
            logger.info(f"search_tool type: {type(self.search_tool)}")
            logger.info(f"search_tool methods: {dir(self.search_tool)}")
            return False
        
        # Check if kernel has the plugin registered
        plugin_registered = False
        
        # Determine the type of kernel.plugins
        if hasattr(self.kernel.plugins, "has_plugin"):
            # New style API
            plugin_registered = self.kernel.plugins.has_plugin("AzureAISearch")
        else:
            # Old style API - plugins might be a dict
            plugin_registered = "AzureAISearch" in self.kernel.plugins
        
        if not plugin_registered:
            logger.error("AzureAISearch plugin not registered with kernel")
            # Try to register it
            try:
                if hasattr(self.kernel.plugins, "add_plugin"):
                    # New style API
                    self.kernel.plugins.add_plugin("AzureAISearch", self.search_tool)
                else:
                    # Old style API - plugins might be a dict
                    self.kernel.plugins["AzureAISearch"] = self.search_tool
                    
                logger.info("AzureAISearch plugin successfully registered with kernel")
            except Exception as e:
                logger.error(f"Failed to register plugin: {e}")
                return False
        
        # Complete the search_internal_documents method
        logger.info("Search plugin configuration verified successfully")
        return True
    
    # Add methods to use the search tool
    async def search_internal_documents(self, query):
        if hasattr(self, "search_tool") and self.search_tool:
            # Use the search tool
            # Implementation depends on how AzureAISearchTool works
            pass
        else:
            return "Search capability not available"
    
    # Add a method to directly access search functionality
    async def search_internal_documents(self, query: str, index_name: str = "sanctiondata-index"):
        """Direct search method using the registered search tool."""
        if not hasattr(self, 'search_tool') or not self.search_tool:
            return "Search tool not available"
        
        try:
            results = await self.search_tool.search(query=query, index_name=index_name)
            return results
        except Exception as e:
            logging.error(f"Error in direct search: {e}")
            return f"Search error: {str(e)}"