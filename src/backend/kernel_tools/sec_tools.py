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
from helpers.summarizeutils import summarize, summarizeTopic
from helpers.analyzers import *
from helpers.reports import ReportLabUtils
from helpers.charting import ReportChartUtils
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from helpers.azureblob import azureBlobApi
import uuid

class SecTools:

    formatting_instructions = "Instructions: returning the output of this function call verbatim to the user in markdown."
    businessOverview = None
    riskAssessment = None
    marketPosition = None
    incomeStatement = None
    incomeSummarization = None
    segmentStatement = None

    agent_name = AgentType.SEC.value

    # Define Company Analyst tools (functions)
    @staticmethod
    @kernel_function(description="analyze the company description for a company from the SEC report")
    async def analyze_company_description(ticker_symbol:str, year:str) -> str:
        global marketPosition
        companyDesc = ReportAnalysisUtils.analyze_company_description(ticker_symbol, year)
        marketPosition = summarize(companyDesc)
        return (
            f"##### Company Description\n"
            f"**Company Name:** {ticker_symbol}\n"
            f"**Company Analysis:** {marketPosition}\n"
            f"{SecTools.formatting_instructions}"
        )

    @staticmethod
    @kernel_function(description="analyze the business highlights for a company from the SEC report")
    async def analyze_business_highlights(ticker_symbol:str, year:str) -> str:
        global businessOverview
        businessHighlights = ReportAnalysisUtils.analyze_business_highlights(ticker_symbol, year)
        businessOverview = summarize(businessHighlights)
        return (
            f"##### Business Highlights\n"
            f"**Company Name:** {ticker_symbol}\n"
            f"**Business Highlights:** {businessOverview}\n"
            f"{SecTools.formatting_instructions}"
        )

    @staticmethod
    @kernel_function(description="analyze the competitors analysis for a company from the SEC report")
    async def get_competitors_analysis(ticker_symbol:str, year:str) -> str:
        compAnalysis = ReportAnalysisUtils.get_competitors_analysis(ticker_symbol, year)
        summarized = summarize(compAnalysis)
        return (
            f"##### Competitor Analysis\n"
            f"**Company Name:** {ticker_symbol}\n"
            f"**Competitor Analysis:** {summarized}\n"
            f"{SecTools.formatting_instructions}"
        )

    @staticmethod
    @kernel_function(description="analyze the risk assessment for a company from the SEC report")
    async def get_risk_assessment(ticker_symbol:str, year:str) -> str:
        global riskAssessment
        riskAssess = ReportAnalysisUtils.get_risk_assessment(ticker_symbol, year)
        riskAssessment = summarize(riskAssess)
        return (
            f"##### Risk Assessment\n"
            f"**Company Name:** {ticker_symbol}\n"
            f"**Risk Assessment Analysis:** {riskAssessment}\n"
            f"{SecTools.formatting_instructions}"
        )

    @staticmethod
    @kernel_function(description="analyze the segment statement for a company from the SEC report")
    async def analyze_segment_stmt(ticker_symbol:str, year:str) -> str:
        global segmentStatement
        segmentStmt = ReportAnalysisUtils.analyze_segment_stmt(ticker_symbol, year)
        segmentStatement = summarize(segmentStmt)
        return (
            f"##### Segment Statement\n"
            f"**Company Name:** {ticker_symbol}\n"
            f"**Segment Statement Analysis:** {segmentStatement}\n"
            f"{SecTools.formatting_instructions}"
        )

    @staticmethod
    @kernel_function(description="analyze the cash flow for a company from the SEC report")
    async def analyze_cash_flow(ticker_symbol:str, year:str) -> str:
        cashFlow = ReportAnalysisUtils.analyze_cash_flow(ticker_symbol, year)
        summarized = summarize(cashFlow)
        return (
            f"##### Cash Flow\n"
            f"**Company Name:** {ticker_symbol}\n"
            f"**Cash Flow Analysis:** {summarized}\n"
            f"{SecTools.formatting_instructions}"
        )

    @staticmethod
    @kernel_function(description="analyze the balance sheet for a company from the SEC report")
    async def analyze_balance_sheet(ticker_symbol:str, year:str) -> str:
        balanceSheet = ReportAnalysisUtils.analyze_balance_sheet(ticker_symbol, year)
        summarized = summarize(balanceSheet)
        return (
            f"##### Balance Sheet\n"
            f"**Company Name:** {ticker_symbol}\n"
            f"**Balance Sheet Analysis:** {summarized}\n"
            f"{SecTools.formatting_instructions}"
        )

    @staticmethod
    @kernel_function(description="analyze the income statement for a company from the SEC report")
    async def analyze_income_stmt(ticker_symbol:str, year:str) -> str:
        global incomeStatement
        incomeStmt = ReportAnalysisUtils.analyze_income_stmt(ticker_symbol, year)
        incomeStatement = summarize(incomeStmt)
        return (
            f"#####Income Statement\n"
            f"**Company Name:** {ticker_symbol}\n"
            f"**Income Statement Analysis:** {incomeStatement}\n"
            f"{SecTools.formatting_instructions}"
        )

    @staticmethod
    @kernel_function(description="analyze the income summarization for a company from the SEC report")
    async def income_summarization(ticker_symbol:str, year:str) -> str:
        global incomeSummarization
        global incomeStatement
        global segmentStatement
        if incomeStatement is None or len(incomeStatement) == 0:
            incomeStmt = ReportAnalysisUtils.analyze_income_stmt(ticker_symbol, year)
            incomeStatement = summarize(incomeStmt)
        if segmentStatement is None or len(segmentStatement) == 0:
            segmentStmt = ReportAnalysisUtils.analyze_segment_stmt(ticker_symbol, year)
            segmentStatement = summarize(segmentStmt)
        incomeSummary = ReportAnalysisUtils.income_summarization(ticker_symbol, year, incomeStatement, segmentStatement)
        incomeSummarization = summarize(incomeSummary)
        return (
            f"#####Income Statement\n"
            f"**Company Name:** {ticker_symbol}\n"
            f"**Income Statement Analysis:** {incomeSummarization}\n"
            f"{SecTools.formatting_instructions}"
        )

    @staticmethod
    @kernel_function(description="build the annual report for a company from the SEC report")
    async def build_annual_report(ticker_symbol:str, year:str) -> str:
        global businessOverview
        global riskAssessment
        global marketPosition
        global incomeSummarization
        if businessOverview is None or len(businessOverview) == 0:
            businessHighlights = ReportAnalysisUtils.analyze_business_highlights(ticker_symbol, year)
            businessOverview = summarize(businessHighlights)
        
        if riskAssessment is None or len(riskAssessment) == 0:
            riskAssess = ReportAnalysisUtils.get_risk_assessment(ticker_symbol, year)
            riskAssessment = summarize(riskAssess)

        if marketPosition is None or len(marketPosition) == 0:
            companyDesc = ReportAnalysisUtils.analyze_company_description(ticker_symbol, year)
            marketPosition = summarize(companyDesc)

        if incomeSummarization is None or len(incomeSummarization) == 0:
            incomeSummary = await SecTools.income_summarization(ticker_symbol, year)
            incomeSummarization = summarize(incomeSummary)
        
        secReport = fmpUtils.get_sec_report(ticker_symbol, year)
        if secReport.find("Date: ") > 0:
            index = secReport.find("Date: ")
            filingDate = secReport[index:].split()[1]
        else:
            filingDate = datetime.now()

        #Convert filing date to datetime and then convert to a formatted string
        if isinstance(filingDate, datetime):
            filingDate = filingDate.strftime("%Y-%m-%d")
        else:
            filingDate = datetime.strptime(filingDate, "%Y-%m-%d").strftime("%Y-%m-%d")


        if Config.APP_IN_CONTAINER:
            reportDir = "/app/backend/reports/"
        else:
            reportDir = "reports\\"
        
        print("****************")
        print("reportDir: ", reportDir)
        print("****************")

        reportFile = reportDir + "{}_Equity_Research_report.pdf".format(ticker_symbol)
        reportFileStock = reportDir + "stock_performance.png"
        reportFilePE = reportDir + "pe_performance.png"
        blobFileName = "{}_{}Equity_Research_report.pdf".format(str(uuid.uuid4()), ticker_symbol)

        ReportChartUtils.get_share_performance(ticker_symbol, filingDate, reportDir)
        ReportChartUtils.get_pe_eps_performance(ticker_symbol, filingDate, 4, reportDir)
        reportOut = ReportLabUtils.build_annual_report(ticker_symbol, reportDir, incomeSummarization,
                                marketPosition, businessOverview, riskAssessment, None, reportFileStock, reportFilePE, filingDate)
        
        try:
            blobUrl = azureBlobApi.copyReport(reportFile, blobFileName)
        except Exception as e:
            reportFile = "/app/backend/reports/" + "{}_Equity_Research_report.pdf".format(ticker_symbol)
            blobUrl = azureBlobApi.copyReport(reportFile, blobFileName)
        
        return (
            f"#####Build Annual Report\n"
            f"**Company Name:** {ticker_symbol}\n"
            f"**Report Saved at :** {blobUrl}\n"
            f"{SecTools.formatting_instructions}"
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
