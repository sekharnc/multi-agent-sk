from typing import List, Optional

import semantic_kernel as sk
from context.cosmos_memory_kernel import CosmosMemoryContext
from kernel_agents.agent_base import BaseAgent
from kernel_tools.forecaster_tools import ForecasterTools
from models.messages_kernel import AgentType
from semantic_kernel.functions import KernelFunction

class ForecasterAgent(BaseAgent):
    """Forecaster agent implementation using Semantic Kernel.

    This agent specializes in all information on company tasks.
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = AgentType.FORECASTER.value,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the Forecast Agent.

        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "CompanyAnalystAgent")
            client: Optional client instance
            definition: Optional definition instance
        """
        # Load configuration if tools not provided
        if not tools:
            # Get tools directly from ForecasterTools class
            tools_dict = ForecasterTools.get_all_kernel_functions()
            tools = [KernelFunction.from_method(func) for func in tools_dict.values()]

            # Use system message from config if not explicitly provided
        if not system_message:
            system_message = self.default_system_message(agent_name)

        # Use agent name from config if available
        agent_name = AgentType.FORECASTER.value

        super().__init__(
            agent_name=agent_name,
            session_id=session_id,
            user_id=user_id,
            memory_store=memory_store,
            tools=tools,
            system_message=system_message,
            client=client,
            definition=definition,
        )

    @staticmethod
    def default_system_message(agent_name=None) -> str:
        """Get the default system message for the agent.
        Args:
            agent_name: The name of the agent (optional)
        Returns:
            The default system message for the agent
        """
        return f"""
            You are a Forecaster and Analysis Agent. 
            Your role is to interpret the output of an extended technical & fundamental analysis pipeline 
            and additional data from the list of one or more the following:
            - Business Overview
            - Risk Assessment
            - Market Position
            - Income Statement
            - Segment Statement
            - Income Summarization
            - Competitor Analysis
            - Business Highlights
            - Business Information
            - Earnings Call Transcripts
            - SEC Reports
            - Analyst Reports
            - News
            - Stock Price Data
            Produce a final recommendation (Buy, Sell, or Hold) with 
            a structured format and thorough, bullet-pointed explanation. 
            You must mention the final probability, interpret it as confidence level, 
            and provide disclaimers like "Past performance is not indicative of future results.
            """

    @property
    def plugins(self):
        """Get the plugins for the Forecastor agent."""
        return ForecasterTools.get_all_kernel_functions()