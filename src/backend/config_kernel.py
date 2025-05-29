# config_kernel.py
import os
import logging
import semantic_kernel as sk
from semantic_kernel.kernel import Kernel

# Updated imports for compatibility
try:
    # Try newer structure
    from semantic_kernel.contents import ChatHistory
except ImportError:
    # Fall back to older structure for compatibility
    from semantic_kernel.connectors.ai.chat_completion_client import ChatHistory
from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent
from azure.ai.projects.models import BingGroundingTool  

# Import AppConfig from app_config
from app_config import config


# This file is left as a lightweight wrapper around AppConfig for backward compatibility
# All configuration is now handled by AppConfig in app_config.py
class Config:
    # Use values from AppConfig
    AZURE_TENANT_ID = config.AZURE_TENANT_ID
    AZURE_CLIENT_ID = config.AZURE_CLIENT_ID
    AZURE_CLIENT_SECRET = config.AZURE_CLIENT_SECRET

    # CosmosDB settings
    COSMOSDB_ENDPOINT = config.COSMOSDB_ENDPOINT
    COSMOSDB_DATABASE = config.COSMOSDB_DATABASE
    COSMOSDB_CONTAINER = config.COSMOSDB_CONTAINER

    # Azure OpenAI settings
    AZURE_OPENAI_DEPLOYMENT_NAME = config.AZURE_OPENAI_DEPLOYMENT_NAME
    AZURE_OPENAI_API_VERSION = config.AZURE_OPENAI_API_VERSION
    AZURE_OPENAI_ENDPOINT = config.AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_SCOPES = config.AZURE_OPENAI_SCOPES

    # Other settings
    FRONTEND_SITE_NAME = config.FRONTEND_SITE_NAME
    AZURE_AI_SUBSCRIPTION_ID = config.AZURE_AI_SUBSCRIPTION_ID
    AZURE_AI_RESOURCE_GROUP = config.AZURE_AI_RESOURCE_GROUP
    AZURE_AI_PROJECT_NAME = config.AZURE_AI_PROJECT_NAME
    AZURE_AI_AGENT_PROJECT_CONNECTION_STRING = (
        config.AZURE_AI_AGENT_PROJECT_CONNECTION_STRING
    )

    @staticmethod
    def GetAzureCredentials():
        """Get Azure credentials using the AppConfig implementation."""
        return config.get_azure_credentials()

    @staticmethod
    def GetCosmosDatabaseClient():
        """Get a Cosmos DB client using the AppConfig implementation."""
        return config.get_cosmos_database_client()

    @staticmethod
    def CreateKernel():
        """Creates a new Semantic Kernel instance using the AppConfig implementation."""
        return config.create_kernel()

    @staticmethod
    def GetAIProjectClient():
        """Get an AIProjectClient using the AppConfig implementation."""
        return config.get_ai_project_client()
    
    @staticmethod
    def GetBingTool() -> BingGroundingTool:
        """Get the BingGroundingTool using the AppConfig implementation."""
        return config.get_bing_tool()
