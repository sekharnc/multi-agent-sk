from typing import List, Optional

import semantic_kernel as sk
from context.cosmos_memory_kernel import CosmosMemoryContext
from kernel_agents.agent_base import BaseAgent
from kernel_tools.technical_tools import TechnicalAnalysisTools
from models.messages_kernel import AgentType
from semantic_kernel.functions import KernelFunction

class TechnicalAnalysisAgent(BaseAgent):
    """Technical agent implementation using Semantic Kernel.

    This agent specializes in Technical analysis tasks.
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = AgentType.TECHNICAL.value,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the Technical Agent.

        Args:
            kernel: The semantic kernel instance
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: List of tools available to this agent (optional)
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "FUNDAMENTAL")
            client: Optional client instance
            definition: Optional definition instance
        """
        # Load configuration if tools not provided
        if not tools:
            # Get tools directly from TechnicalAnalysisTools class
            tools_dict = TechnicalAnalysisTools.get_all_kernel_functions()
            tools = [KernelFunction.from_method(func) for func in tools_dict.values()]

            # Use system message from config if not explicitly provided
        if not system_message:
            system_message = self.default_system_message(agent_name)

        # Use agent name from config if available
        agent_name = AgentType.TECHNICAL.value

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
        return """
                You are a specialized Technical Analysis Agent. 
                You have access to historical stock price data and can detect 
                multiple technical strategies, signals, and candlestick patterns
                (EMA crossover, RSI, MACD w/ zero-line checks, Bollinger Bands, 
                Stochastics, ATR, ADX, hammer, engulfing, etc.). 
                Provide a JSON-structured output of your findings, 
                including an overall (naive) rating. 
                Other agents will consume your results.
                """

    @property
    def plugins(self):
        """Get the plugins for the Technical agent."""
        return TechnicalAnalysisTools.get_all_kernel_functions()