"""Factory for creating agents in the Multi-Agent Custom Automation Engine."""

import logging
from typing import Dict, List, Callable, Any, Optional, Type
from types import SimpleNamespace
from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction
from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent
import inspect

from kernel_agents.agent_base import BaseAgent

# Import the new AppConfig instance
from app_config import config

# Import all specialized agent implementations
#from kernel_agents.hr_agent import HrAgent
from kernel_agents.human_agent import HumanAgent
from kernel_agents.company_agent import CompanyAnalystAgent
from kernel_agents.earningcalls_agent import EarningCallsAgent
from kernel_agents.fundamental_agent import FundamentalAnalysisAgent
from kernel_agents.technical_agent import TechnicalAnalysisAgent
from kernel_agents.sec_agent import SecAgent
from kernel_agents.forecaster_agent import ForecasterAgent
from kernel_agents.web_agent import WebAgent  # Import WebAgent for generic web tasks
from kernel_agents.generic_agent import GenericAgent
from kernel_agents.planner_agent import PlannerAgent  # Add PlannerAgent import
from kernel_agents.group_chat_manager import GroupChatManager
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import PlannerResponsePlan, AgentType

from azure.ai.projects.models import (
    ResponseFormatJsonSchema,
    ResponseFormatJsonSchemaType,
)

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating agents in the Multi-Agent Custom Automation Engine."""

    # Mapping of agent types to their implementation classes
    _agent_classes: Dict[AgentType, Type[BaseAgent]] = {
        AgentType.COMPANY: CompanyAnalystAgent,
        AgentType.EARNINGCALLS: EarningCallsAgent,
        AgentType.FUNDAMENTAL: FundamentalAnalysisAgent,
        AgentType.TECHNICAL: TechnicalAnalysisAgent,
        AgentType.SEC: SecAgent,
        AgentType.FORECASTER: ForecasterAgent,
        AgentType.GENERIC: GenericAgent,
        AgentType.WEB: WebAgent,  # Add WebAgent for generic web tasks
        AgentType.HUMAN: HumanAgent,
        AgentType.PLANNER: PlannerAgent,
        AgentType.GROUP_CHAT_MANAGER: GroupChatManager,  # Add GroupChatManager
    }

    # Mapping of agent types to their string identifiers (for automatic tool loading)
    _agent_type_strings: Dict[AgentType, str] = {
        AgentType.COMPANY: AgentType.COMPANY.value,
        AgentType.EARNINGCALLS: AgentType.EARNINGCALLS.value,
        AgentType.FUNDAMENTAL: AgentType.FUNDAMENTAL.value,
        AgentType.TECHNICAL: AgentType.TECHNICAL.value,
        AgentType.SEC: AgentType.SEC.value,
        AgentType.FORECASTER: AgentType.FORECASTER.value,
        AgentType.GENERIC: AgentType.GENERIC.value, 
        AgentType.WEB: AgentType.WEB.value,  # Use WEB for generic web tasks    
        AgentType.HUMAN: AgentType.HUMAN.value,
        AgentType.PLANNER: AgentType.PLANNER.value,
        AgentType.GROUP_CHAT_MANAGER: AgentType.GROUP_CHAT_MANAGER.value,
    }

    # System messages for each agent type
    _agent_system_messages: Dict[AgentType, str] = {
        AgentType.COMPANY: CompanyAnalystAgent.default_system_message(),
        AgentType.EARNINGCALLS: EarningCallsAgent.default_system_message(),
        AgentType.FUNDAMENTAL: FundamentalAnalysisAgent.default_system_message(),
        AgentType.TECHNICAL: TechnicalAnalysisAgent.default_system_message(),
        AgentType.SEC: SecAgent.default_system_message(),
        AgentType.FORECASTER: ForecasterAgent.default_system_message(),
        AgentType.GENERIC: GenericAgent.default_system_message(),
        AgentType.WEB: WebAgent.default_system_message(),
        AgentType.HUMAN: HumanAgent.default_system_message(),
        AgentType.PLANNER: PlannerAgent.default_system_message(),
        AgentType.GROUP_CHAT_MANAGER: GroupChatManager.default_system_message(),
    }

    # Cache of agent instances by session_id and agent_type
    _agent_cache: Dict[str, Dict[AgentType, BaseAgent]] = {}

    # Cache of Azure AI Agent instances
    _azure_ai_agent_cache: Dict[str, Dict[str, AzureAIAgent]] = {}

    @classmethod
    async def create_agent(
        cls,
        agent_type: AgentType,
        session_id: str,
        user_id: str,
        temperature: float = 0.0,
        memory_store: Optional[CosmosMemoryContext] = None,
        system_message: Optional[str] = None,
        response_format: Optional[Any] = None,
        client: Optional[Any] = None,
        **kwargs,
    ) -> BaseAgent:
        """Create an agent of the specified type.

        This method creates and initializes an agent instance of the specified type. If an agent
        of the same type already exists for the session, it returns the cached instance. The method
        handles the complete initialization process including:
        1. Creating a memory store for the agent
        2. Setting up the Semantic Kernel
        3. Loading appropriate tools from JSON configuration files
        4. Creating an Azure AI agent definition using the AI Project client
        5. Initializing the agent with all required parameters
        6. Running any asynchronous initialization if needed
        7. Caching the agent for future use

        Args:
            agent_type: The type of agent to create (from AgentType enum)
            session_id: The unique identifier for the current session
            user_id: The user identifier for the current user
            temperature: The temperature parameter for the agent's responses (0.0-1.0)
            system_message: Optional custom system message to override default
            response_format: Optional response format configuration for structured outputs
            **kwargs: Additional parameters to pass to the agent constructor

        Returns:
            An initialized instance of the specified agent type

        Raises:
            ValueError: If the agent type is unknown or initialization fails
        """
        # Check if we already have an agent in the cache
        if (
            session_id in cls._agent_cache
            and agent_type in cls._agent_cache[session_id]
        ):
            logger.info(
                f"Returning cached agent instance for session {session_id} and agent type {agent_type}"
            )
            return cls._agent_cache[session_id][agent_type]
        
        # Get the agent class
        agent_class = cls._agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Create memory store
        if memory_store is None:
            memory_store = CosmosMemoryContext(session_id, user_id)

        # Use default system message if none provided
        if system_message is None:
            system_message = cls._agent_system_messages.get(
                agent_type,
                f"You are a helpful AI assistant specialized in {cls._agent_type_strings.get(agent_type, 'general')} tasks.",
            )
 

        # For other agent types, use the standard tool loading mechanism
        agent_type_str = cls._agent_type_strings.get(
            agent_type, agent_type.value.lower()
        )
        tools = None

        # Build the agent definition (functions schema)
        definition = None

        try:
            if client is None:
                # Create the AIProjectClient instance using the config
                # This is a placeholder; replace with actual client creation logic
                client = config.get_ai_project_client()
        except Exception as client_exc:
            logger.error(f"Error creating AIProjectClient: {client_exc}")
            raise

        try:
            # Create the agent definition using the AIProjectClient (project-based pattern)
            # For GroupChatManager, create a definition with minimal configuration
            if client is not None:
                agent_id = None
                found_agent = False
                agent_list = await client.agents.list_agents()
                for agent in agent_list.data:
                    if agent.name == agent_type_str:
                        agent_id = agent.id
                        found_agent = True
                        break
                if found_agent:
                    definition = await client.agents.get_agent(agent_id)
                else:
                    definition = await client.agents.create_agent(
                        model=config.AZURE_OPENAI_DEPLOYMENT_NAME,
                        name=agent_type_str,
                        instructions=system_message,
                        temperature=temperature,
                        response_format=response_format,  # Add response_format if required
                    )
                logger.info(
                    f"Successfully created agent definition for {agent_type_str}"
                )
        except Exception as agent_exc:
            logger.error(
                f"Error creating agent definition with AIProjectClient for {agent_type_str}: {agent_exc}"
            )

            raise

        # Create the agent instance using the project-based pattern
        try:
            # Filter kwargs to only those accepted by the agent's __init__
            agent_init_params = inspect.signature(agent_class.__init__).parameters
            valid_keys = set(agent_init_params.keys()) - {"self"}
            filtered_kwargs = {
                k: v
                for k, v in {
                    "agent_name": agent_type_str,
                    "session_id": session_id,
                    "user_id": user_id,
                    "memory_store": memory_store,
                    "tools": tools,
                    "system_message": system_message,
                    "client": client,
                    "definition": definition,
                    **kwargs,
                }.items()
                if k in valid_keys
            }
            agent = agent_class(**filtered_kwargs)

            # Initialize the agent asynchronously if it has async_init
            if hasattr(agent, "async_init") and inspect.iscoroutinefunction(
                agent.async_init
            ):
                init_result = await agent.async_init()

        except Exception as e:
            logger.error(
                f"Error creating agent of type {agent_type} with parameters: {e}"
            )
            raise

        # Cache the agent instance
        if session_id not in cls._agent_cache:
            cls._agent_cache[session_id] = {}
        cls._agent_cache[session_id][agent_type] = agent

        return agent

    @classmethod
    async def create_all_agents(
        cls,
        session_id: str,
        user_id: str,
        temperature: float = 0.0,
        memory_store: Optional[CosmosMemoryContext] = None,
        client: Optional[Any] = None,
    ) -> Dict[AgentType, BaseAgent]:
        """Create all agent types for a session in a specific order.

        This method creates all agent instances for a session in a multi-phase approach:
        1. First, it creates all basic agent types except for the Planner and GroupChatManager
        2. Then it creates the Planner agent, providing it with references to all other agents
        3. Finally, it creates the GroupChatManager with references to all agents including the Planner

        This ordered creation ensures that dependencies between agents are properly established,
        particularly for the Planner and GroupChatManager which need to coordinate other agents.

        Args:
            session_id: The unique identifier for the current session
            user_id: The user identifier for the current user
            temperature: The temperature parameter for agent responses (0.0-1.0)

        Returns:
            Dictionary mapping agent types (from AgentType enum) to initialized agent instances
        """

        # Create each agent type in two phases
        # First, create all agents except PlannerAgent and GroupChatManager
        agents = {}
        planner_agent_type = AgentType.PLANNER
        group_chat_manager_type = AgentType.GROUP_CHAT_MANAGER
        web_agent_type = AgentType.WEB
        

        try:
            if client is None:
                # Create the AIProjectClient instance using the config
                client = config.get_ai_project_client()
        except Exception as client_exc:
            logger.error(f"Error creating AIProjectClient: {client_exc}")
        # Initialize cache for this session if it doesn't exist
        if session_id not in cls._agent_cache:
            cls._agent_cache[session_id] = {}

        # Phase 1: Create all agents except planner and group chat manager
        for agent_type in [
            at
            for at in cls._agent_classes.keys()
            if at != planner_agent_type and at != group_chat_manager_type 
        ]:
            if agent_type == web_agent_type:
                # For web agent type, use the specialized create_web_agent method
                agents[agent_type] = await cls.create_web_agent(
                    session_id=session_id,
                    user_id=user_id,
                    temperature=temperature,
                    client=client,
                    memory_store=memory_store,
                )
            else:
                agents[agent_type] = await cls.create_agent(
                    agent_type=agent_type,
                    session_id=session_id,
                    user_id=user_id,
                    temperature=temperature,
                    client=client,
                    memory_store=memory_store,
                )

        # Create agent name to instance mapping for the planner
        agent_instances = {}
        for agent_type, agent in agents.items():
            agent_name = agent_type.value

            logging.info(
                f"Creating agent instance for {agent_name} with type {agent_type}"
            )
            agent_instances[agent_name] = agent

        # Log the agent instances for debugging
        logger.info(
            f"Created {len(agent_instances)} agent instances for planner: {', '.join(agent_instances.keys())}"
        )
        
        # Phase 2: Create the planner agent with agent_instances
        planner_agent = await cls.create_agent(
            agent_type=AgentType.PLANNER,
            session_id=session_id,
            user_id=user_id,
            temperature=temperature,
            agent_instances=agent_instances,  # Pass agent instances to the planner
            client=client,
            response_format=ResponseFormatJsonSchemaType(
                json_schema=ResponseFormatJsonSchema(
                    name=PlannerResponsePlan.__name__,
                    description=f"respond with {PlannerResponsePlan.__name__.lower()}",
                    schema=PlannerResponsePlan.model_json_schema(),
                )
            ),
        )
        agent_instances[AgentType.PLANNER.value] = (
            planner_agent  # to pass it to group chat manager
        )
        agents[planner_agent_type] = planner_agent

        # Phase 3: Create group chat manager with all agents including the planner
        group_chat_manager = await cls.create_agent(
            agent_type=AgentType.GROUP_CHAT_MANAGER,
            session_id=session_id,
            user_id=user_id,
            temperature=temperature,
            client=client,
            agent_instances=agent_instances,
        )
        agents[group_chat_manager_type] = group_chat_manager

        return agents

    @classmethod
    def get_agent_class(cls, agent_type: AgentType) -> Type[BaseAgent]:
        """Get the agent class for the specified type.

        Args:
            agent_type: The agent type

        Returns:
            The agent class

        Raises:
            ValueError: If the agent type is unknown
        """
        agent_class = cls._agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return agent_class

    @classmethod
    def clear_cache(cls, session_id: Optional[str] = None) -> None:
        """Clear the agent cache.

        Args:
            session_id: If provided, clear only this session's cache
        """
        if session_id:
            if session_id in cls._agent_cache:
                del cls._agent_cache[session_id]
                logger.info(f"Cleared agent cache for session {session_id}")
            if session_id in cls._azure_ai_agent_cache:
                del cls._azure_ai_agent_cache[session_id]
                logger.info(f"Cleared Azure AI agent cache for session {session_id}")
        else:
            cls._agent_cache.clear()
            cls._azure_ai_agent_cache.clear()
            logger.info("Cleared all agent caches")

    @classmethod
    async def create_web_agent(
        cls,
        session_id: str,
        user_id: str,
        temperature: float = 0.0,
        memory_store: Optional[CosmosMemoryContext] = None,
        system_message: Optional[str] = None,
        response_format: Optional[Any] = None,
        client: Optional[Any] = None,
        **kwargs,
    ) -> BaseAgent:
        """Create a specialized web agent.

        This method creates and initializes a web agent instance with specific configurations
        for web-based tasks. It follows a similar initialization process as create_agent but
        with web-specific optimizations.

        Args:
            session_id: The unique identifier for the current session
            user_id: The user identifier for the current user
            temperature: The temperature parameter for the agent's responses (0.0-1.0)
            system_message: Optional custom system message to override default
            response_format: Optional response format configuration for structured outputs
            **kwargs: Additional parameters to pass to the agent constructor

        Returns:
            An initialized instance of the web agent

        Raises:
            ValueError: If initialization fails
        """
        agent_type = AgentType.WEB
        
        # Check if we already have an agent in the cache
        if (
            session_id in cls._agent_cache
            and agent_type in cls._agent_cache[session_id]
        ):
            logger.info(
                f"Returning cached web agent instance for session {session_id}"
            )
            return cls._agent_cache[session_id][agent_type]
        
        # Get the agent class
        agent_class = cls._agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(f"Web agent type not found")

        # Create memory store
        if memory_store is None:
            memory_store = CosmosMemoryContext(session_id, user_id)

        # Use default system message if none provided
        if system_message is None:
            system_message = cls._agent_system_messages.get(
                agent_type,
                """You are a specialized web assistant focused on gathering and processing information from online sources.
                You have access to the Bing search tool. ALWAYS use this tool when you need to:
                1. Search for current information
                2. Find specific facts or data
                3. Research topics or questions
                4. Answer queries that require up-to-date information
                
                DO NOT try to answer questions requiring external information without using the Bing search tool first.
                If you're asked to find information online, ALWAYS use the Bing search tool before responding.
                """
            )

        # For web agent, use the standard tool loading mechanism
        agent_type_str = cls._agent_type_strings.get(
            agent_type, agent_type.value.lower()
        )
        tools = None

        # Import and set up BingGroundingTool
        try:
            bing_tool = await config.get_bing_tool()
        except Exception as bing_exc:
            logger.error(f"Error setting up BingGroundingTool: {bing_exc}")     
            bing_tool = None

        # Build the agent definition (functions schema)
        definition = None

        try:
            if client is None:
                # Create the AIProjectClient instance using the config
                client = config.get_ai_project_client()
        except Exception as client_exc:
            logger.error(f"Error creating AIProjectClient for web agent: {client_exc}")
            raise

        try:
            # Create the agent definition using the AIProjectClient (project-based pattern)
            if client is not None:
                agent_id = None
                found_agent = False
                agent_list = await client.agents.list_agents()
                for agent in agent_list.data:
                    if agent.name == agent_type_str:
                        agent_id = agent.id
                        found_agent = True
                        break
                if found_agent:
                    definition = await client.agents.get_agent(agent_id)
                else:
                    definition = await client.agents.create_agent(
                        model=config.AZURE_OPENAI_DEPLOYMENT_NAME,
                        name=agent_type_str,
                        instructions=system_message,
                        temperature=temperature,
                        response_format=response_format,
                    )
                
                # If we have the Bing tool, add it to the agent
                if bing_tool and definition:
                    try:
                        # Register the Bing grounding tool with the agent definition
                        tool_def = bing_tool.definitions
                        
                        # More detailed logging for debugging tool structure
                        logger.info(f"Available methods on client.agents: {[method for method in dir(client.agents) if not method.startswith('_')]}")
                        logger.info(f"Tool definition structure: {type(tool_def)}")
                        
                        # Check if tool_def is properly structured
                        if isinstance(tool_def, list):
                            logger.info(f"Tool definition is a list with {len(tool_def)} items")
                            for i, tool in enumerate(tool_def):
                                logger.info(f"Tool {i} type: {type(tool)}, keys: {tool.keys() if hasattr(tool, 'keys') else 'N/A'}")
                            
                            # Ensure the Bing tool is configured correctly
                            for tool in tool_def:
                                if hasattr(tool, 'get') and tool.get('name') == 'bing_search':
                                    logger.info("Found Bing search tool in definition")
                                    break
                            else:
                                logger.warning("Bing search tool not found in tool definitions")
                            
                            # Update the agent with the tool list
                            await client.agents.update_agent(
                                agent_id=definition.id,
                                tools=tool_def  # Pass the list directly
                            )
                            logger.info(f"Updated agent with tools")
                        else:
                            logger.info(f"Tool definition is not a list, wrapping in list: {tool_def}")
                            await client.agents.update_agent(
                                agent_id=definition.id,
                                tools=[tool_def]  # Wrap in a list if it's a single object
                            )
                        
                        # Make an explicit verification call to ensure tool was added
                        updated_agent = await client.agents.get_agent(definition.id)
                        logger.info(f"Verified agent tools: {getattr(updated_agent, 'tools', 'No tools attribute')}")
                        logger.info(f"Added BingGroundingTool to web agent {agent_type_str}")
                    except Exception as tool_exc:
                        logger.error(f"Error adding BingGroundingTool to agent: {tool_exc}")
                        import traceback
                        logger.error(f"Detailed error: {traceback.format_exc()}")
                    
                logger.info(
                    f"Successfully created web agent definition for {agent_type_str}"
                )
        except Exception as agent_exc:
            logger.error(
                f"Error creating web agent definition with AIProjectClient: {agent_exc}"
            )
            raise

        # Create the web agent instance
        try:
            # Filter kwargs to only those accepted by the agent's __init__
            agent_init_params = inspect.signature(agent_class.__init__).parameters
            valid_keys = set(agent_init_params.keys()) - {"self"}
            filtered_kwargs = {
                k: v
                for k, v in {
                    "agent_name": agent_type_str,
                    "session_id": session_id,
                    "user_id": user_id,
                    "memory_store": memory_store,
                    "tools": tools,
                    "system_message": system_message,
                    "client": client,
                    "definition": definition,
                    "bing_tool": bing_tool,  # Pass the Bing tool to the agent
                    **kwargs,
                }.items()
                if k in valid_keys
            }
            agent = agent_class(**filtered_kwargs)

            # Initialize the agent asynchronously if it has async_init
            if hasattr(agent, "async_init") and inspect.iscoroutinefunction(
                agent.async_init
            ):
                init_result = await agent.async_init()

        except Exception as e:
            logger.error(
                f"Error creating web agent: {e}"
            )
            raise

        # Cache the agent instance
        if session_id not in cls._agent_cache:
            cls._agent_cache[session_id] = {}
        cls._agent_cache[session_id][agent_type] = agent

        return agent
