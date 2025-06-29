import inspect
import json
from datetime import datetime
from typing import Annotated, Callable, List, Optional
import logging
logger = logging.getLogger(__name__)

from semantic_kernel.functions import kernel_function
from typing import Any, Dict, List, get_type_hints
from models.messages_kernel import AgentType

class WebTools:
    """Define Web Agent functions (tools) for KYC-related company information gathering"""
    formatting_instructions = """
    Instructions for formatting search results:
    
    1. Organize all search results in a clear markdown structure
    2. Use headers and bullet points for readability
    3. Always include a Sources section with URLs
    4. Format company name as an H4 header
    5. Format section titles as H5 headers
    6. Bold all key data points found
    
    Example format:
    
    #### [Company Name] Information
    
    ##### [Section Title]
    - **[Data Point Label]:** [Value found]
    - **[Data Point Label]:** [Value found]
    
    ##### Sources
    - [Source name 1]: [URL]
    - [Source name 2]: [URL]
    """
    
    agent_name = AgentType.WEB.value

    @staticmethod
    @kernel_function(description="Get company identity information including legal name, ownership structure, and official registered address.")
    async def get_company_identity_info(
        company_name: Annotated[str, "The name of the company to research"]
    ) -> str:
        """Get company identity information for KYC verification.
        
        Retrieves the legal identity information and address details needed to verify
        the company's existence and registration status.
        
        Args:
            company_name: The name of the company to research
            
        Returns:
            Information about company ownership, legal name and official address
        """
        print(f"FUNCTION CALLED: get_company_identity_info for company: {company_name}")
        logger.info(f"get_company_identity_info called for company: {company_name}")
        result = f"""**SEARCH REQUEST**: Use bing_search tool to find comprehensive address and ownership information for {company_name}

        **Search Queries to Use:**
        1. "{company_name}" + "company address" + "headquarters" + "registered address" + "business address"
        2. "{company_name}" + "ownership type" + "private public" + "legal name" + "official name"


        **Required Information to Find:**
        - Ownership Type: Determine if this is Private, Public, or Government Sponsored Entity
        - Legal Name: The official legal name of the entity
        - Address: The complete registered business address
        - Address Type: Specify if this is a Company address or Individual address

        {WebTools.formatting_instructions}

        Please search for this information using web search and provide citations with URLs for all sources found."""
        
        logger.info(f"get_company_identity_info completed for company: {company_name}")
        return result
#         logger.info(f"get_company_identity_info completed for company: {company_name}")

#     @staticmethod
#     @kernel_function(description="Get general business information for a company including public trading status, legal entity details, and financial information.")
#     async def get_business_info(
#         company_name: Annotated[str, "The name of the company to research for business information"]
#     ) -> str:
#         """Get comprehensive business information for KYC purposes."""
#         return f"""**SEARCH REQUEST**: Use Bing web search to find comprehensive business information for {company_name}

# **Search Queries to Use:**
# 1. "{company_name}" + "publicly traded" + "stock ticker" + "NAICS code" + "industry classification"
# 2. "{company_name}" + "annual revenue" + "financial information" + "country incorporation" + "headquarters location" + "legal entity type" + "corporation"


# **Required Information to Find:**
# - Publicly Traded: Yes/No
# - Stock Ticker: (if publicly traded)
# - Exchange Listed On: (if applicable - NYSE, NASDAQ, etc.)
# - NAICS Code: North American Industry Classification System code
# - Legal Entity is a Non-Operating Entity: Yes/No
# - Type of Non-Operating entity: Disregarded Entity, Special Purpose Entity, Special Purpose Vehicle, Other, or Not Applicable
# - Legal Entity Type: Corporation, Government Entity, Individual, or Sole Proprietor
# - Country of Incorporation: Where the entity was legally formed
# - Country of Headquarters: Where the main operations are based
# - Estimated Annual Revenue: Latest available revenue figures

# {WebTools.formatting_instructions}

# Please search for this information using web search and provide citations with URLs for all sources found."""

#     @staticmethod
#     @kernel_function(description="Get information about the legal entity's source of funds and wealth for KYC compliance.")
#     async def get_funds_wealth_info(
#         company_name: Annotated[str, "The name of the company to research for source of funds and wealth information"]
#     ) -> str:
#         """Get source of funds and wealth information for KYC purposes.
        
#         This function will search for information about how the legal entity 
#         generates revenue and accumulates wealth.
        
#         Args:
#             company_name: The name of the company to research
            
#         Returns:
#             Information about the entity's source of funds and wealth
#         """
#         return f"""Please search for information about {company_name}'s source of funds and wealth and provide details on:

# **Required Information:**
# - Primary Revenue Sources: How the company generates income
# - Business Model: Description of how the company operates and makes money
# - Major Clients/Customers: Key sources of revenue (if publicly available)
# - Investment Sources: Where the company gets its funding (investors, loans, etc.)
# - Asset Base: Types of assets the company holds
# - Financial History: Track record of revenue generation and wealth accumulation

# {WebTools.formatting_instructions}

# Please search annual reports, SEC filings, investor presentations, business news, and financial databases. Focus on legitimate business activities and revenue streams. Provide citations for all sources."""

#     @staticmethod
#     @kernel_function(description="Get detailed business activity information to determine what specific activities the customer is engaged in from a predefined list."
#     )
#     async def get_activity_details(
#         company_name: Annotated[str, "The name of the company to research for business activities"]
#     ) -> str:
#         """Get detailed business activity information for KYC risk assessment.
        
#         This function will search for and identify which activities the customer 
#         is engaged in from a specific list of business categories.
        
#         Args:
#             company_name: The name of the company to research
            
#         Returns:
#             List of business activities the company is engaged in from the predefined categories
#         """
#         business_activities = """
#         Accountant, Aircraft Dealer, Arctic Activity, Arts or Antiques, ATM Owner/Servicer, 
#         Auto Parts/Repair/Service, Auto Title Lending, Auto/Truck Dealer, Bearer Shares, 
#         Beer/Wine/Liquor Store, Boat/Motor Home Dealer, Casino/Gaming Industry, CBD, 
#         Check Casher, Church/Mosque/Synagogue/Temple, Cigarette Distributor, Cleaning Service, 
#         Computer Services, Construction/Plumbing/HVAC, Consulting Services, Consumer Lender, 
#         Convenience Store/Gas Station, Debt Collector, Deposit Brokers, Educational Services, 
#         Event Planning and Management, Farm/Farmer's Market, FinTech, For. Embassy/Consul/Mission, 
#         Foreign Shell Bank, Gold Exch/Coins/Precious Metals, Grocery/Supermarket, 
#         Group Foundation/Trust, Guns/Weapons/Ammunition, Healthcare and Wellness, Hemp, 
#         Import/Export/Shipping, Internet Gambling, Internet Sweepstakes Cafes/Gaming Centers, 
#         Internet/E-Commerce, Investment/Securities/Broker, Issuing or selling of traveler's checks or money orders, 
#         Jewelry/Gems, Landscaper, Lawyer/Legal Services, Leasing/Property Management, 
#         Marijuana/Ancillary Marijuana, Marketing/Media, Medical Doctor/Dentist, 
#         Money Exchange/Funds Transfer, Mountaintop Removal Activities, Online Pharmacy, 
#         Parking Garage, Pawnshop, Payday Lender, Photography, Political Party/Campaign, 
#         Private Prison, Professional Service Providers, Providing or selling of prepaid access, 
#         Real Estate, Restaurants/Bars/Grills, Retail Store, Salon/Beauty/Spas, Scrap Metal Dealer, 
#         Security Services, Sport and Fitness, Tax Refund Company, Telemarketing, 
#         Third-Party Payment/Payroll Processors, Travel Agency, Troop/Scout Org, 
#         Trucking and Transportation, Vending Machine Operator, Virtual Currency, None of the above
#         """
        
#         return f"""Please search for detailed information about {company_name}'s business activities and operations. Based on your findings, identify which of the following activities the company is engaged in:

# **Business Activity Categories:**
# {business_activities}

# **Required Information:**
# - Primary Business Activities: Main activities the company engages in
# - Secondary Activities: Any additional business lines or services
# - Industry Classification: How the company classifies itself
# - Licenses and Permits: Any special licenses that indicate specific activities
# - Products and Services: Detailed description of what the company offers

# {WebTools.formatting_instructions}

# Please search the company website, business registrations, regulatory filings, industry databases, and news sources. Match the company's activities to the specific categories listed above. If the company engages in multiple activities, list all applicable ones. Provide citations for all sources."""

    @staticmethod
    @kernel_function(description="Get comprehensive financial and business profile including revenue sources, business model, and financial status.")
    async def get_financial_business_profile(
        company_name: Annotated[str, "The name of the company to research"]
    ) -> str:
        """Get detailed financial and business profile for KYC risk assessment.
                Retrieves comprehensive information about the company's business model, 
        financial status, revenue sources, and industry classification for
        thorough KYC risk evaluation.
        
        Args:
            company_name: The name of the company to research
            
        Returns:
            Comprehensive business and financial profile information
        """
        logger.info(f"get_financial_business_profile called for company: {company_name}")
        return f"""**SEARCH REQUEST**: Use bing_search to find a comprehensive financial and business profile for {company_name}

**Search Queries to Use:**
1. "{company_name}" + "annual revenue" + "business model" + "financial information"
2. "{company_name}" + "funding sources" + "investors" + "revenue streams"
3. "{company_name}" + "industry classification" + "NAICS code" + "publicly traded"

**Required Information to Find:**
- Business Structure: 
  - Publicly Traded: Yes/No
  - Stock Ticker & Exchange (if applicable)
  - Legal Entity Type
  - Country of Incorporation & Headquarters
  
- Financial Profile:
  - Estimated Annual Revenue
  - Primary Revenue Sources
  - Business Model Description
  - Major Clients/Customers
  - Investment/Funding Sources
  - Asset Base
  
- Industry Information:
  - NAICS Code/Industry Classification
  - Primary Business Sector

{WebTools.formatting_instructions}

Please search annual reports, SEC filings, investor presentations, business news, and financial databases. Provide citations for all sources."""

    @staticmethod
    @kernel_function(description="Identify specific high-risk or regulated business activities the company is engaged in from a predefined list for KYC risk categorization.")
    async def get_regulated_activity_details(
        company_name: Annotated[str, "The name of the company to research"]
    ) -> str:
        """Identify regulated or high-risk business activities for KYC risk categorization.
        
        This function identifies which specific regulated, high-risk, or special 
        business categories the company belongs to from a predefined list used
        for KYC risk scoring and regulatory compliance.
        
        Args:
            company_name: The name of the company to research
            
        Returns:
            Identified regulated or high-risk business activities
        """
        logger.info(f"get_regulated_activity_details called for company: {company_name}")
        return f"""**SEARCH REQUEST**: Use bing_search to find about {company_name}'s business activities and operations. Based on your findings, identify which of the following regulated or high-risk activities the company is engaged in:

**Regulated/High-Risk Business Activity Categories:**
- Money Services Business (MSB)
- Currency Exchange
- Virtual Currency Exchange
- Prepaid Access Programs
- Stored Value Facilities
- Electronic Money Issuers
- Payment Processing Services
- Crowdfunding Platforms
- Peer-to-Peer Lending
- Factoring
- Asset Management
- Investment Advisory
- Underwriting
- Credit Rating Agencies
- Insurance
- Gambling
- Real Estate
- Precious Metals and Stones Dealers
- Art Dealers
- Auction Houses
- Notaries
- Trust and Company Service Providers
- High-Value Goods Dealers
- Other (specify)

**Required Information:**
- Primary Regulated Activities: Main regulated activities the company engages in
- Secondary Regulated Activities: Any additional regulated business lines or services
- Industry Classification: How the company classifies itself
- Licenses and Permits: Any special licenses that indicate specific regulated activities
- Products and Services: Detailed description of what the company offers

{WebTools.formatting_instructions}

Please search the company website, business registrations, regulatory filings, industry databases, and news sources. Match the company's activities to the specific regulated categories listed above. If the company engages in multiple activities, list all applicable ones. Provide citations for all sources."""

    @classmethod
    def get_all_kernel_functions(cls) -> dict[str, Callable]:
        """
        Returns a dictionary of all methods in this class that have the @kernel_function annotation.
        
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
            if hasattr(method, "__kernel_function__"):
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
                    # Skip first parameter 'cls' for class methods
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
                    "agent": cls.agent_name,
                    "function": name,
                    "description": description,
                    "arguments": json.dumps(args_dict).replace('"', "'"),
                }

                tools_list.append(tool_entry)

        # Return the JSON string representation
        return json.dumps(tools_list, ensure_ascii=False, indent=2)
