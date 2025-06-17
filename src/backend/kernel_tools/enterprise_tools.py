import inspect
import time
import logging
from datetime import datetime
from typing import Annotated, Callable, List

from semantic_kernel.functions import kernel_function
from models.messages_kernel import AgentType
import inspect
import json
from typing import Any, Dict, List, get_type_hints
logger = logging.getLogger(__name__)

class EnterpriseTools:
    """Define Generic Agent functions (tools)"""

    agent_name = AgentType.ENTERPRISE.value
    # @staticmethod
    # @kernel_function(description="This function retrieves sanctions and risk category details of a country using Azure AI Search RAG process.")
    # async def get_internal_risk_details(
    #     country_name: Annotated[str, "The name of the country to search for sanction and risk category details"]
    # ) -> str:
    #     """This function checks sanctions risk category."""
    #     return f"""**SEARCH REQUEST**: Use AzureAISearch to find sanctions and risk category details for {country_name}

    # **Search Query to Use:**
    # "country sanctions risk category {country_name}"

    # **Required Information to Find:**
    # - Sanctions Status: Whether the country is currently sanctioned
    # - Risk Category: The risk classification of the country
    
    # Please search internal documents using AzureAISearch and provide all relevant information with citations."""

    @staticmethod
    @kernel_function(description="This function retrieves sanctions and risk category details of a country using Azure AI Search RAG process.")
    async def get_internal_risk_details(
        country_name: Annotated[str, "The name of the country to search for sanction and risk category details"]
    ) -> str:
        """This function checks sanctions risk category using AzureAISearch on sanctionsdata-index."""
        try:
            # Note: This function doesn't directly access Azure Search
            # The actual search will be performed by AzureAISearchTool when called through the agent
            # We just need to provide the search query and instructions
            search_query = f"country sanctions risk category {country_name}"
            logger.info(f"Preparing to RAG search for sanctions and risk category details for {country_name} with query: {search_query}")
            return f"""**EXECUTING SEARCH**: Searching for sanctions and risk category details for {country_name}

**Using Azure AI Search on sanctionsdata-index with query:**
"{search_query}"

**Looking for:**
- Sanctions Status
- Risk Category

This request should be executed by the AzureAISearchTool."""
        except Exception as e:
            logging.error(f"Error in get_internal_risk_details: {str(e)}")
            return f"Error searching for {country_name}: {str(e)}"

    @staticmethod
    @kernel_function(description="Directly search the sanctionsdata-index using Azure AI Search")
    async def search_sanctions_data(
        query: Annotated[str, "The search query to use for finding sanctions information"],
        index_name: Annotated[str, "The index name to search in (default: sanctionsdata-index)"] = "sanctionsdata-index"
    ) -> str:
        """
        Directly search the sanctions data index using Azure AI Search.
        This function requires the AzureAISearchTool to be available in the kernel.
        """
        # This function doesn't actually perform the search itself
        # It's meant to be used by the agent which will invoke the AzureAISearchTool
        logger.info(f"Preparing to DIRECT search in index '{index_name}' with query: {query}")
        return f"""**DIRECT SEARCH REQUEST**:
        
**Index:** {index_name}
**Query:** {query}

Please execute this search using the AzureAISearchTool and format the results."""

    @classmethod
    def get_all_kernel_functions(cls) -> dict[str, Callable]:
        """
        Returns a dictionary of all methods in this class that have the @kernel_function annotation.
        This function itself is not annotated with @kernel_function.

        Returns:
            Dict[str, Callable]: Dictionary with function names as keys and function objects as values
        """
        kernel_functions = {}

        # Get all class methods
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            # Skip this method itself and any private/special methods
            if name.startswith("_") or name == "get_all_kernel_functions":
                continue

            # Check if the method has the kernel_function annotation
            # by looking at its __annotations__ attribute
            method_attrs = getattr(method, "__annotations__", {})
            if hasattr(method, "__kernel_function__") or "kernel_function" in str(
                method_attrs
            ):
                kernel_functions[name] = method

        return kernel_functions

    @classmethod
    def generate_tools_json_doc(cls) -> str:
        """
        Generate a JSON document containing information about all methods in the class.

        Returns:
            str: JSON string containing the methods' information
        """

        tools_list = []

        # Get all methods from the class that have the kernel_function annotation
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            # Skip this method itself and any private methods
            if name.startswith("_") or name == "generate_tools_json_doc":
                continue

            # Check if the method has the kernel_function annotation
            if hasattr(method, "__kernel_function__"):
                # Get method description from docstring or kernel_function description
                description = ""
                if hasattr(method, "__doc__") and method.__doc__:
                    description = method.__doc__.strip()

                # Get kernel_function description if available
                if hasattr(method, "__kernel_function__") and getattr(
                    method.__kernel_function__, "description", None
                ):
                    description = method.__kernel_function__.description

                # Get argument information by introspection
                sig = inspect.signature(method)
                args_dict = {}

                # Get type hints if available
                type_hints = get_type_hints(method)

                # Process parameters
                for param_name, param in sig.parameters.items():
                    # Skip first parameter 'cls' for class methods (though we're using staticmethod now)
                    if param_name in ["cls", "self"]:
                        continue

                    # Get parameter type
                    param_type = "string"  # Default type
                    if param_name in type_hints:
                        type_obj = type_hints[param_name]
                        # Convert type to string representation
                        if hasattr(type_obj, "__name__"):
                            param_type = type_obj.__name__.lower()
                        else:
                            # Handle complex types like List, Dict, etc.
                            param_type = str(type_obj).lower()
                            if "int" in param_type:
                                param_type = "int"
                            elif "float" in param_type:
                                param_type = "float"
                            elif "bool" in param_type:
                                param_type = "boolean"
                            else:
                                param_type = "string"

                    # Create parameter description
                    param_desc = param_name.replace("_", " ")
                    args_dict[param_name] = {
                        "description": param_name,
                        "title": param_name.replace("_", " ").title(),
                        "type": param_type,
                    }

                # Add the tool information to the list
                tool_entry = {
                    "agent": cls.agent_name,  # Use HR agent type
                    "function": name,
                    "description": description,
                    "arguments": json.dumps(args_dict).replace('"', "'"),
                }

                tools_list.append(tool_entry)

        # Return the JSON string representation
        return json.dumps(tools_list, ensure_ascii=False, indent=2)
