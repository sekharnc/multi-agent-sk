import inspect
import time
import os
import logging
from datetime import datetime
from typing import Annotated, Callable, List

from semantic_kernel.functions import kernel_function
from models.messages_kernel import AgentType
import inspect
import json
from typing import Any, Dict, List, Optional, get_type_hints
from azure.core.exceptions import HttpResponseError
from app_config import config

logger = logging.getLogger(__name__)

class EnterpriseTools:
    """Define Generic Agent functions (tools)"""
    formatting_instructions = """
    Instructions for formatting search results:
    
    1. Organize all search results in a clear markdown structure
    2. Use headers and bullet points for readability
    3. Always include a Sources section with URLs
    4. Format company name as an H4 header
    5. Format section titles as H5 headers
    6. Bold all key data points found
    
    Example format:
    
    #### [Country Name] Information
    
    ##### [Section Title]
    - **[Data Point Label]:** [Value found]
    - **[Data Point Label]:** [Value found]
    
    ##### Sources
    - [Source name 1]: [URL]
    - [Source name 2]: [URL]
    """
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
        """This function checks AI Search knowledgebase for sanctions risk category using AzureAISearch."""
        try:
            # Note: This function doesn't directly access Azure Search
            # The actual search will be performed by AzureAISearchTool when called through the agent
            # We just need to provide the search query and instructions
            search_query = f"find risk category details for {country_name}"
            logger.info(f"Preparing to RAG search for sanctions and risk category details for {country_name} with query: {search_query}")
            results = await EnterpriseTools.search_knowledge_base(
                query=search_query)
            if not results:
                return f"No results found for {country_name}. Please check the country name or try a different query."
            return results
        except HttpResponseError as e:
            logging.error(f"Error in get_internal_risk_details: {str(e)}")
            return f"Error searching for {country_name}: {str(e)}"  
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

    @staticmethod
    @kernel_function(description="Search the AI Search Knowledge base for the given query.")
    async def search_knowledge_base(
        query: str,
        top: int = 3,
        filter: Optional[str] = None
    ) -> str:
        """
        Search the AI Search Knowledge base for the given query.
        This function requires the AzureAISearchTool to be available in the kernel.

        Args:
            query (str): The search query to use.
            top (int): The number of top results to return. Default is 3.
            filter (Optional[str]): Optional filter expression to apply to the search results.

        Returns:
            JSON string containing the search results.
        """
        logger.info(f"Preparing to search knowledge base with query: {query}, top: {top}, filter: {filter}")
        try:
            # get the search client
            search_client = await config.get_azure_search_client()
            if not search_client:
                return json.dumps({
                    "success": False,
                    "error": "Azure Search client is not configured properly.",
                    "results": []
                })
            # Perform the search
            results = search_client.search(
                search_text=query,
                # top=top,
                # select=["chunk_id","chunk", "title", "url", "filepath"],
                # # Include other fields as needed
                # filter=filter
            )
            # Process the results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "chunk_id": result.get("chunk_id") if "chunk_id" in result else None,
                    "title": result.get("title") if "title" in result else None,
                    "chunk": result.get("chunk") if "chunk" in result else None,
                    "url": result.get("url") if "url" in result else None,
                    "filepath": result.get("filepath") if "filepath" in result else None,
                    "score": result["@search.score"] if "@search.score" in result else None
                }
                formatted_results.append(formatted_result)
            # Return the results as a JSON string
            return json.dumps({
                "success": True,
                "query": query,
                "results": formatted_results
            }, indent=2) + F"\n\n{EnterpriseTools.formatting_instructions}"
        except HttpResponseError as e:
            logging.error(f"Error during search: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "results": []
            })
        except Exception as e:
            logging.error(f"Unexpected error during search: {str(e)}")
            return json.dumps({
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "results": []
            })

    @staticmethod
    @kernel_function(description="Search for files containing the query text.")
    async def file_search(
        query: str,
        file_types: str = "pdf,md,txt",
        max_results: int = 5
    ) -> str:
        """
        Search for files containing the query text.
        
        Args:
            query (str): The search query to use.
            file_types (str): Comma-separated list of file types to search in (default: pdf,md,txt).
            max_results (int): Maximum number of results to return (default: 5).

        Returns:
            JSON string containing matching file paths and content snippets.
        """
        logger.info(f"Preparing to search files with query: {query}, file_types: {file_types}, max_results: {max_results}")
        if not config.FILE_SEARCH_ENABLED:
            return json.dumps({
                "success": False,
                "error": "File search is not enabled in the configuration.",
                "results": []
            })
        search_path = config.FILE_SEARCH_PATH
        if not search_path or not os.path.exists(search_path):
            return json.dumps({
                "success": False,
                "error": f"File search path '{search_path}' does not exist or is not configured.",
                "results": []
            })
        
        try:
            # parse file types to include
            extensions = [ext.strip().lower() for ext in file_types.split(",") if ext.strip()]
            results = []
            # Walk through the directory and search for files
            for root, _, files in os.walk(search_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if query.lower() in content.lower():
                                    # find the context of the match (snippet)
                                    lower_content = content.lower()
                                    match_index = lower_content.find(query.lower())
                                    start_index = max(0, match_index - 100)  # 100 chars before
                                    end_index = min(len(content), match_index + 100)  # 100 chars
                                    snippet = content[start_index:end_index]
                                    if start_index > 0:
                                        snippet = "..." + snippet
                                    if end_index < len(content):
                                        snippet += "..."
                                    # Append the result
                                    results.append({
                                        "filepath": file_path,
                                        "filename": file,
                                        "snippet": snippet                                        
                                    })
                                    if len(results) >= max_results:
                                        break
                        except Exception as e:
                            logging.error(f"Error reading file {file_path}: {str(e)}")
                if len(results) >= max_results:
                    break
            # Return the results as a JSON string
            return json.dumps({
                "success": True,
                "query": query,
                "results": results
            }, indent=2) + F"\n\n{EnterpriseTools.formatting_instructions}"
        except Exception as e:
            logging.error(f"Error during file search: {str(e)}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "results": []
            })
        
    @staticmethod
    @kernel_function(description="Get information about the configured knowledge base.")
    async def get_knowledge_base_info() -> str:
        """
        Get information about the configured knowledge base.
        
        Returns:
            JSON string containing knowledge base configuration details.
        """
        logger.info("Preparing to get knowledge base info")
        
        ai_search_enabled = config.AI_SEARCH_ENABLED
        file_search_enabled = config.FILE_SEARCH_ENABLED
        info = {
            "ai_search": {
                "enabled": ai_search_enabled,
                "index": config.AI_SEARCH_INDEX if ai_search_enabled else None,
                "endpoint": config.AI_SEARCH_ENDPOINT if ai_search_enabled else None
            },
            "file_search": {
                "enabled": file_search_enabled,
                "path": config.FILE_SEARCH_PATH if file_search_enabled else None
            }
        }
        return json.dumps(info, indent=2) + F"\n\n{EnterpriseTools.formatting_instructions}"
    
    
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
        Generate a JSON document for Enterprise Agent.

        Returns:
            Dictionary containing the tools JSON document.
        
        """
        return {
            "agent": cls.agent_name,
            "tools": [
                {
                    "name": "get_internal_risk_details",
                    "description": "Retrieve sanctions and risk category details of a country using Azure AI Search RAG process.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "country_name": {
                                "type": "string",
                                "description": "The name of the country to search for sanction and risk category details"
                            }
                        },
                        "required": ["country_name"]
                    }
                },
                {
                    "name": "search_sanctions_data",
                    "description": "Directly search the sanctionsdata-index using Azure AI Search.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to use for finding sanctions information"
                            },
                            "index_name": {
                                "type": "string",
                                "default": "sanctionsdata-index",
                                "description": "The index name to search in"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "search_knowledge_base",
                    "description": "Search the AI Search Knowledge base for the given query.",
                    "parameters": {
                        "query": {
                            "type": "string",
                            "description": "The search query to use."
                        },
                        "top": {
                            "type": "integer",
                            "default": 3,
                            "description": "The number of top results to return. Default is 3."
                        },
                        "filter": {
                            "type": "string",
                            "description": ("Optional filter expression to apply to the search results. "
                                            "This can be used to filter results based on specific criteria.")
                        }

                    }
                },
                {
                    "name": "file_search",
                    "description": "Search for files containing the query text.",
                    "parameters": {
                        "query": {
                            "type": "string",
                            "description": "The search query to use."
                        },
                        "file_types": {
                            "type": "string",
                            "description": "Comma-separated list of file types to search in (default: pdf,md,txt)."
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum number of results to return (default: 5)."
                        }
                    }
                },
                {
                    "name": "get_knowledge_base_info",
                    "description": "Get information about the configured knowledge base.",
                    "parameters": {}
                }
            ]
            
        }
    # @classmethod
    # def generate_tools_json_doc(cls) -> str:
    #     """
    #     Generate a JSON document containing information about all methods in the class.

    #     Returns:
    #         str: JSON string containing the methods' information
    #     """

    #     tools_list = []

    #     # Get all methods from the class that have the kernel_function annotation
    #     for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
    #         # Skip this method itself and any private methods
    #         if name.startswith("_") or name == "generate_tools_json_doc":
    #             continue

    #         # Check if the method has the kernel_function annotation
    #         if hasattr(method, "__kernel_function__"):
    #             # Get method description from docstring or kernel_function description
    #             description = ""
    #             if hasattr(method, "__doc__") and method.__doc__:
    #                 description = method.__doc__.strip()

    #             # Get kernel_function description if available
    #             if hasattr(method, "__kernel_function__") and getattr(
    #                 method.__kernel_function__, "description", None
    #             ):
    #                 description = method.__kernel_function__.description

    #             # Get argument information by introspection
    #             sig = inspect.signature(method)
    #             args_dict = {}

    #             # Get type hints if available
    #             type_hints = get_type_hints(method)

    #             # Process parameters
    #             for param_name, param in sig.parameters.items():
    #                 # Skip first parameter 'cls' for class methods (though we're using staticmethod now)
    #                 if param_name in ["cls", "self"]:
    #                     continue

    #                 # Get parameter type
    #                 param_type = "string"  # Default type
    #                 if param_name in type_hints:
    #                     type_obj = type_hints[param_name]
    #                     # Convert type to string representation
    #                     if hasattr(type_obj, "__name__"):
    #                         param_type = type_obj.__name__.lower()
    #                     else:
    #                         # Handle complex types like List, Dict, etc.
    #                         param_type = str(type_obj).lower()
    #                         if "int" in param_type:
    #                             param_type = "int"
    #                         elif "float" in param_type:
    #                             param_type = "float"
    #                         elif "bool" in param_type:
    #                             param_type = "boolean"
    #                         else:
    #                             param_type = "string"

    #                 # Create parameter description
    #                 param_desc = param_name.replace("_", " ")
    #                 args_dict[param_name] = {
    #                     "description": param_name,
    #                     "title": param_name.replace("_", " ").title(),
    #                     "type": param_type,
    #                 }

    #             # Add the tool information to the list
    #             tool_entry = {
    #                 "agent": cls.agent_name,  # Use HR agent type
    #                 "function": name,
    #                 "description": description,
    #                 "arguments": json.dumps(args_dict).replace('"', "'"),
    #             }

    #             tools_list.append(tool_entry)

    #     # Return the JSON string representation
    #     return json.dumps(tools_list, ensure_ascii=False, indent=2)
