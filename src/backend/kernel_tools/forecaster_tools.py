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

class ForecasterTools:

    agent_name = AgentType.FORECASTER.value

    # Define Company Analyst tools (functions)
    @staticmethod
    @kernel_function(description="Interprets the JSON output from ExtendedCombinedAnalysisAgent. "
                "Generates a final Buy/Sell/Hold recommendation with a structured rationale, "
                "risk factors, disclaimers, and an explanation of the probability or confidence.")
    async def analyze_and_predict(analysis_result: Dict[str, Any]) -> str:
        """
        Takes the JSON output from ExtendedCombinedAnalysisAgent (technical indicators,
        candlestick patterns, fundamentals, news sentiment, final decision),
        and uses an LLM to produce a structured forecast with:
        1) A multi-section format (Introduction, Technical, Fundamental, etc.)
        2) An explanation of probability/score as confidence (e.g., 70% => "moderately strong")
        3) A final recommendation
        4) Legal disclaimers

        Returns a markdown or text response with these structured sections.
        """
        # Convert analysis_result into a JSON string
        analysis_json_str = json.dumps(analysis_result, indent=2)

        # Extract the final probability from the JSON for prompt usage
        final_decision = analysis_result.get("final_decision", {})
        probability_value = final_decision.get("probability", None)
        rating_value = final_decision.get("rating", "hold")

        # We can provide instructions to interpret the confidence level:
        # e.g., 0.0-0.33 => "low confidence", 0.33-0.66 => "moderate confidence", 0.66-1.0 => "high confidence"
        # We'll do a bit of logic to embed in the prompt. Alternatively, let the LLM do it entirely.
        confidence_descriptor = "moderate"
        if probability_value is not None:
            if probability_value <= 0.33:
                confidence_descriptor = "low"
            elif probability_value >= 0.66:
                confidence_descriptor = "high"
            else:
                confidence_descriptor = "moderate"

        # Construct a detailed prompt with strict output structure
        prompt = f"""
        You are a specialized financial analysis LLM. You have received a JSON structure that
        represents an extended analysis of a stock, including:
        - Technical signals (RSI, MACD, Bollinger, EMA crossover, Stochastics, ADX)
        - Candlestick pattern detections (TA-Lib)
        - Basic fundamentals (P/E ratios, etc.)
        - News sentiment
        - A final numeric probability (score) and rating (Buy/Sell/Hold).

        The JSON data is:

        ```
        {analysis_json_str}
        ```

        **Please return your answer in the following sections:**

        1) **Introduction**
        - Briefly introduce the analysis.

        2) **Technical Overview**
        - Summarize the key technical indicators and any candlestick patterns.
        - Explain whether they are bullish, bearish, or neutral.

        3) **Fundamental Overview**
        - Mention any notable fundamental data (like forwardPE, trailingPE, etc.) 
            and how it influences the outlook.

        4) **News & Sentiment**
        - Highlight the sentiment score (range: -1.0 to +1.0). 
            Explain if it's a tailwind (positive) or headwind (negative).

        5) **Probability & Confidence**
        - The system’s final probability is **{probability_value}** (range: 0.0 to 1.0).
        - Interpret it as **{confidence_descriptor}** confidence 
            (e.g., <=0.33 => "low", 0.33-0.66 => "moderate", >=0.66 => "high").
        - Elaborate how confident or uncertain this rating might be based on
            conflicting signals, volatility, etc.

        6) **Final Recommendation**
        - Based on the system’s final rating: **{rating_value}**.
        - Explain briefly why you agree or disagree, or how you interpret it.

        7) **Disclaimers**
        - Include disclaimers such as "Past performance is not indicative of future results."
        - Remind the user that this is not guaranteed investment advice.
        - Encourage further research before making any decisions.

        Please format your response in **Markdown**, with headings for each section
        and bullet points where appropriate. 
        """

        return prompt

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
