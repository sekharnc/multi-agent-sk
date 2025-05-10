import inspect
from typing import Annotated, Callable, List, Dict

from semantic_kernel.functions import kernel_function
from models.messages_kernel import AgentType
import inspect
import json
from typing import Any, Dict, List, get_type_hints
from helpers.fmputils import *
from helpers.yfutils import *
from datetime import date, timedelta, datetime

class CompanyAnalystTools:

    formatting_instructions = "Instructions: returning the output of this function call verbatim to the user in markdown. Then write AGENT SUMMARY: and then include a summary of what you did."
    agent_name = AgentType.COMPANY.value

    # Define Company Analyst tools (functions)
    @staticmethod
    @kernel_function(description="get a company's profile information")
    async def get_company_info(ticker_symbol: str) -> str:
        return (
            f"##### Get Company Information\n"
            f"**Company Name:** {ticker_symbol}\n"
            f"**Company Information:** {fmpUtils.get_company_profile(ticker_symbol)}\n"
            f"{CompanyAnalystTools.formatting_instructions}"
        )

    @staticmethod
    @kernel_function(description="get analyst recommendation for a designated company")
    async def get_analyst_recommendations(ticker_symbol: str) -> str:
        return (
            f"##### Get Company Recommendations\n"
            f"**Company Name:** {ticker_symbol}\n"
            f"**Recommendations:** {yfUtils.get_analyst_recommendations(ticker_symbol)}\n"
            f"{CompanyAnalystTools.formatting_instructions}"
        )
    
    @staticmethod
    @kernel_function(description="retrieve stock price data for designated ticker symbol")
    async def get_stock_data(ticker_symbol: str) -> str:
        end_date = date.today().strftime("%Y-%m-%d")
        start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        return (
            f"##### Stock Data from Yahoo Finance\n"
            f"**Company Name:** {ticker_symbol}\n\n"
            f"**Start Date:** {start_date}\n"
            f"**End Date:** {end_date}\n\n"
            f"**Stock Data:** {yfUtils.get_stock_data(ticker_symbol, start_date, end_date)}\n"
            f"{CompanyAnalystTools.formatting_instructions}"
        )

    @staticmethod
    @kernel_function(description="get latest financial basics for a designated company")
    async def get_financial_metrics(ticker_symbol: str) -> str:
        years = 4
        return (
            f"##### Get Financial Information\n"
            f"**Company Name:** {ticker_symbol}\n\n"
            f"**Years:** {years}\n\n"
            f"**Financial Information:** {fmpUtils.get_financial_metrics(ticker_symbol, years)}\n"
            f"{CompanyAnalystTools.formatting_instructions}"
        )

    @staticmethod
    @kernel_function(description="retrieve market news related to designated company")
    async def get_company_news(ticker_symbol: str) -> str:
        end_date = date.today().strftime("%Y-%m-%d")
        start_date = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
        return (
            f"##### Get Company News\n"
            f"**Company Name:** {ticker_symbol}\n\n"
            #f"**Company News:** {fmpUtils.get_company_news(ticker_symbol, start_date, end_date)}\n"
            f"**Company News:** {yfUtils.get_company_news(ticker_symbol, start_date, end_date)}\n"
            f"{CompanyAnalystTools.formatting_instructions}"
        )

    @staticmethod
    @kernel_function(description="Analyze the data that you have access to like news and analyst recommendations and provide a sentiment analysis, positive or negative outlook")
    async def get_sentiment_analysis(ticker_symbol: str) -> str:
        return (
            f"##### Get Company Information\n"
            f"**Company Name:** {ticker_symbol}\n"
            f"{CompanyAnalystTools.formatting_instructions}"
        )

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

    # This function does NOT have the kernel_function annotation
    # because it's meant for introspection rather than being exposed as a tool
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
