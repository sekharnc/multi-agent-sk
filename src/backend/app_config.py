# app_config.py
import os
import logging
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.cosmos.aio import CosmosClient
from azure.ai.projects.aio import AIProjectClient
from semantic_kernel.kernel import Kernel
from semantic_kernel.contents import ChatHistory
from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent
from semantic_kernel.functions import KernelFunction
from azure.monitor.opentelemetry import configure_azure_monitor
from azure.ai.projects.models import BingGroundingTool
from azure.ai.projects.models import AzureAISearchTool  


# Load environment variables from .env file
load_dotenv()


class AppConfig:
    """Application configuration class that loads settings from environment variables."""

    def __init__(self):
        """Initialize the application configuration with environment variables."""
        # Azure authentication settings
        self.AZURE_TENANT_ID = self._get_optional("AZURE_TENANT_ID")
        self.AZURE_CLIENT_ID = self._get_optional("AZURE_CLIENT_ID")
        self.AZURE_CLIENT_SECRET = self._get_optional("AZURE_CLIENT_SECRET")

        # CosmosDB settings
        self.COSMOSDB_ENDPOINT = self._get_optional("COSMOSDB_ENDPOINT")
        self.COSMOSDB_DATABASE = self._get_optional("COSMOSDB_DATABASE")
        self.COSMOSDB_CONTAINER = self._get_optional("COSMOSDB_CONTAINER")

        # Azure OpenAI settings
        self.AZURE_OPENAI_DEPLOYMENT_NAME = self._get_required(
            "AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"
        )
        self.AZURE_OPENAI_API_VERSION = self._get_required(
            "AZURE_OPENAI_API_VERSION", "2024-11-20"
        )
        self.AZURE_OPENAI_ENDPOINT = self._get_required("AZURE_OPENAI_ENDPOINT")
        self.AZURE_OPENAI_SCOPES = [
            f"{self._get_optional('AZURE_OPENAI_SCOPE', 'https://cognitiveservices.azure.com/.default')}"
        ]

        # Frontend settings
        self.FRONTEND_SITE_NAME = self._get_optional(
            "FRONTEND_SITE_NAME", "http://127.0.0.1:3000"
        )

        self.AZURE_BLOB_STORAGE_NAME = self._get_required("AZURE_BLOB_STORAGE_NAME")
        self.AZURE_BLOB_CONTAINER_NAME = self._get_required("AZURE_BLOB_CONTAINER_NAME")

        self.BING_CONNECTION_NAME_ENV = self._get_required("BING_CONNECTION_NAME_ENV")

        self.AZURE_AI_SEARCH_INDEX_NAME = self._get_required(
            "AZURE_AI_SEARCH_INDEX_NAME", "sanctionsdata-index"
        )

        # Azure AI settings
        self.AZURE_AI_SUBSCRIPTION_ID = self._get_required("AZURE_AI_SUBSCRIPTION_ID")
        self.AZURE_AI_RESOURCE_GROUP = self._get_required("AZURE_AI_RESOURCE_GROUP")
        self.AZURE_AI_PROJECT_NAME = self._get_required("AZURE_AI_PROJECT_NAME")
        self.AZURE_AI_AGENT_PROJECT_CONNECTION_STRING = self._get_required(
            "AZURE_AI_AGENT_PROJECT_CONNECTION_STRING"
        )

        self.APPLICATIONINSIGHTS_CONNECTION_STRING = self._get_optional(
            "APPLICATIONINSIGHTS_CONNECTION_STRING", ""
        )

        # Cached clients and resources
        self._azure_credentials = None
        self._cosmos_client = None
        self._cosmos_database = None
        self._ai_project_client = None

    def _get_required(self, name: str, default: Optional[str] = None) -> str:
        """Get a required configuration value from environment variables.

        Args:
            name: The name of the environment variable
            default: Optional default value if not found

        Returns:
            The value of the environment variable or default if provided

        Raises:
            ValueError: If the environment variable is not found and no default is provided
        """
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            logging.warning(
                "Environment variable %s not found, using default value", name
            )
            return default
        raise ValueError(
            f"Environment variable {name} not found and no default provided"
        )

    def _get_optional(self, name: str, default: str = "") -> str:
        """Get an optional configuration value from environment variables.

        Args:
            name: The name of the environment variable
            default: Default value if not found (default: "")

        Returns:
            The value of the environment variable or the default value
        """
        if name in os.environ:
            return os.environ[name]
        return default

    def _get_bool(self, name: str) -> bool:
        """Get a boolean configuration value from environment variables.

        Args:
            name: The name of the environment variable

        Returns:
            True if the environment variable exists and is set to 'true' or '1', False otherwise
        """
        return name in os.environ and os.environ[name].lower() in ["true", "1"]

    def get_azure_credentials(self):
        """Get Azure credentials using DefaultAzureCredential.

        Returns:
            DefaultAzureCredential instance for Azure authentication
        """
        # Cache the credentials object
        if self._azure_credentials is not None:
            print("Using cached Azure credentials")
            logging.info("Using cached Azure credentials")
            return self._azure_credentials

        try:
            self._azure_credentials = DefaultAzureCredential()
            logging.info("Created new Azure credentials")
            print("Using Default Azure credentials")
            logging.info("Using Default Azure credentials")
            return self._azure_credentials
        except Exception as exc:
            logging.warning("Failed to create DefaultAzureCredential: %s", exc)
            return None

    def get_cosmos_database_client(self):
        """Get a Cosmos DB client for the configured database.

        Returns:
            A Cosmos DB database client
        """
        try:
            if self._cosmos_client is None:
                self._cosmos_client = CosmosClient(
                    self.COSMOSDB_ENDPOINT, credential=self.get_azure_credentials()
                )

            if self._cosmos_database is None:
                self._cosmos_database = self._cosmos_client.get_database_client(
                    self.COSMOSDB_DATABASE
                )

            return self._cosmos_database
        except Exception as exc:
            logging.error(
                "Failed to create CosmosDB client: %s. CosmosDB is required for this application.",
                exc,
            )
            raise

    def create_kernel(self):
        """Creates a new Semantic Kernel instance.

        Returns:
            A new Semantic Kernel instance
        """
        # Create a new kernel instance without manually configuring OpenAI services
        # The agents will be created using Azure AI Agent Project pattern instead
        kernel = Kernel()
        return kernel

    def get_ai_project_client(self):
        """Create and return an AIProjectClient for Azure AI Foundry using from_connection_string.

        Returns:
            An AIProjectClient instance
        """
        if self._ai_project_client is not None:
            return self._ai_project_client

        try:
            credential = self.get_azure_credentials()
            if credential is None:
                raise RuntimeError(
                    "Unable to acquire Azure credentials; ensure DefaultAzureCredential is configured"
                )

            connection_string = self.AZURE_AI_AGENT_PROJECT_CONNECTION_STRING
            self._ai_project_client = AIProjectClient.from_connection_string(
                credential=credential, conn_str=connection_string
            )

            ## Enable the AI tracing
            # application_insights_connection_string = self.APPLICATIONINSIGHTS_CONNECTION_STRING
            # logging.info(
            #     "Application Insights connection string: %s",
            #     application_insights_connection_string,
            # )
            # if not application_insights_connection_string:
            #     logging.info("Application Insights was not enabled for this project.")
            #     logging.info("Enable it via the 'Tracing' tab in your AI Foundry project page.")
            # configure_azure_monitor(connection_string=application_insights_connection_string)

            # # enable additional instrumentations if needed
            # self._ai_project_client.telemetry.enable()


            return self._ai_project_client
        except Exception as exc:
            logging.error("Failed to create AIProjectClient: %s", exc)
            raise
    async def get_bing_tool(self) -> 'BingGroundingTool':
        """Get the BingGroundingTool using the AIProjectClient.

        Returns:
            An instance of BingGroundingTool or None if there's an error
        """
        try:
            client = self.get_ai_project_client()
            bing_connection_name = self.BING_CONNECTION_NAME_ENV
            logging.info(f"Getting Bing connection with name: {bing_connection_name}")
            
            # Check if we have a valid connection name
            if not bing_connection_name:
                logging.error("BING_CONNECTION_NAME_ENV is not set or is empty")
                return None
                
            try:
                bing_connection = await client.connections.get(connection_name=bing_connection_name)
                logging.info(f"Successfully retrieved Bing connection: {bing_connection.id}")
                
                bing_tool = BingGroundingTool(connection_id=bing_connection.id)
                logging.info("Successfully created BingGroundingTool")
                
                return bing_tool
            except Exception as conn_exc:
                logging.error(f"Error getting Bing connection: {conn_exc}")
                import traceback
                logging.error(f"Detailed connection error: {traceback.format_exc()}")
                return None
                
        except Exception as exc:
            logging.error(f"Failed to get BingGroundingTool: {exc}")
            import traceback
            logging.error(f"Detailed BingGroundingTool error: {traceback.format_exc()}")
            return None
    async def get_azure_ai_search_tool(self) -> 'AzureAISearchTool':
        """Get the AzureAISearchTool using the AIProjectClient.

        Returns:
            An instance of AzureAISearchTool or None if there's an error
        """
        try:
            client = self.get_ai_project_client()
            search_index_name = self.AZURE_AI_SEARCH_INDEX_NAME
            logging.info(f"Getting Azure AI Search index with name: {search_index_name}")

            # Check if we have a valid index name
            if not search_index_name:
                logging.error("AZURE_AI_SEARCH_INDEX_NAME is not set or is empty")
                return None

            try:
                # Look for a connection with azureAISearch type similar to how we get the Bing connection
                conn_list = await client.connections.list()
                for conn in conn_list:
                    logging.info(f"Checking connection: {conn.id} of type {conn.connection_type}")
                    if conn.connection_type == "CognitiveSearch":
                        logging.info(f"Found Azure AI Search connection: {conn.id}")
                        # Try creating the tool using parameters from the docs
                        # We may need to adjust parameters based on the exact SDK version
                        try:
                            # First attempt - using properties we know from BingGroundingTool
                            search_tool = AzureAISearchTool(
                                index_connection_id=conn.id,
                                index_name=search_index_name
                            )
                            logging.info("Successfully created AzureAISearchTool with connection_id and index_name")
                            return search_tool
                        except TypeError as e:
                            logging.warning(f"First attempt failed: {e}, trying alternative parameters...")
                            try:
                                # Second attempt - using just index_name
                                search_tool = AzureAISearchTool(
                                    index_name=search_index_name
                                )
                                logging.info("Successfully created AzureAISearchTool with just index_name")
                                return search_tool
                            except TypeError:
                                # Third attempt - no parameters
                                search_tool = AzureAISearchTool()
                                logging.info("Created basic AzureAISearchTool without parameters")
                                return search_tool

                # These statements should be outside the loop to execute only after checking all connections
                logging.warning("No Azure AI Search connection found")
                return None
                
            except Exception as search_exc:
                logging.error(f"Error getting Azure AI Search tool: {search_exc}")
                import traceback
                logging.error(f"Detailed search tool error: {traceback.format_exc()}")
                return None

        except Exception as exc:
            logging.error(f"Failed to get AzureAISearchTool: {exc}")
            import traceback
            logging.error(f"Detailed AzureAISearchTool error: {traceback.format_exc()}")
            return None
    async def create_azure_ai_agent(
        self,
        agent_name: str,
        instructions: str,
        tools: Optional[List[KernelFunction]] = None,
        client=None,
        response_format=None,
        temperature: float = 0.0,
    ):
        """
        Creates a new Azure AI Agent with the specified name and instructions using AIProjectClient.
        If an agent with the given name (assistant_id) already exists, it tries to retrieve it first.

        Args:
            kernel: The Semantic Kernel instance
            agent_name: The name of the agent (will be used as assistant_id)
            instructions: The system message / instructions for the agent
            agent_type: The type of agent (defaults to "assistant")
            tools: Optional tool definitions for the agent
            tool_resources: Optional tool resources required by the tools
            response_format: Optional response format to control structured output
            temperature: The temperature setting for the agent (defaults to 0.0)

        Returns:
            A new AzureAIAgent instance
        """
        try:
            # Get the AIProjectClient
            if client is None:
                client = self.get_ai_project_client()

            # # ToDo: This is the fixed code but commenting it out as agent clean up is no happening yet
            #  # and there are multiple versions of agents due to testing
            # # First try to get an existing agent with this name as assistant_id
            # try:
            #     agent_id = None
            #     agent_list = await client.agents.list_agents()
            #     for agent in agent_list.data:
            #         if agent.name == agent_name:
            #             agent_id = agent.id
            #             break
            #             # If the agent already exists, we can use it directly
            #             # Get the existing agent definition
            #     existing_definition = await client.agents.get_agent(agent_id)
            #     # Create the agent instance directly with project_client and existing definition
            #     agent = AzureAIAgent(
            #         client=client,
            #         definition=existing_definition,
            #         plugins=tools,
            #     )

            #     client.agents.list_agents()

            #     return agent
            # except Exception as e:
            #     # The Azure AI Projects SDK throws an exception when the agent doesn't exist
            #     # (not returning None), so we catch it and proceed to create a new agent
            #     if "ResourceNotFound" in str(e) or "404" in str(e):
            #         logging.info(
            #             f"Agent with ID {agent_name} not found. Will create a new one."
            #         )
            #     else:
            #         # Log unexpected errors but still try to create a new agent
            #         logging.warning(
            #             f"Unexpected error while retrieving agent {agent_name}: {str(e)}. Attempting to create new agent."
            #         )

            agent_id = None
            found_agent = False
            agent_list = await client.agents.list_agents()
            for agent in agent_list.data:
                if agent.name == agent_name:
                    agent_id = agent.id
                    found_agent = True
                    break
            if found_agent:
                agent_definition = await client.agents.get_agent(agent_id)
            else:
                # Create the agent using the project client with the agent_name as both name and assistantId
                agent_definition = await client.agents.create_agent(
                    model=self.AZURE_OPENAI_DEPLOYMENT_NAME,
                    name=agent_name,
                    instructions=instructions,
                    temperature=temperature,
                    response_format=response_format,
                )

            # Create the agent instance directly with project_client and definition
            agent = AzureAIAgent(
                client=client,
                definition=agent_definition,
                plugins=tools,
            )

            return agent
        except Exception as exc:
            logging.error("Failed to create Azure AI Agent: %s", exc)
            raise

    def get_azure_ai_search_tool_sync(self) -> 'AzureAISearchTool':
        """Synchronous version of get_azure_ai_search_tool.
        
        Returns:
            An instance of AzureAISearchTool or None if there's an error
        """
        try:
            import asyncio
            # Use the event loop if available or create a new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context already
                    return asyncio.create_task(self.get_azure_ai_search_tool())
                else:
                    # We're in a sync context
                    return loop.run_until_complete(self.get_azure_ai_search_tool())
            except RuntimeError:
                # No event loop available, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.get_azure_ai_search_tool())
                loop.close()
                return result
        except Exception as e:
            logging.error(f"Error in get_azure_ai_search_tool_sync: {e}")
            return None


# Create a global instance of AppConfig
config = AppConfig()
