import logging
import uuid
import json
import re
import datetime
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from azure.ai.projects.models import (
    ResponseFormatJsonSchema,
    ResponseFormatJsonSchemaType,
)
import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.agents import (
    AzureAIAgent,
    AzureAIAgentSettings,
    AzureAIAgentThread,
)
from kernel_agents.agent_base import BaseAgent
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    AgentMessage,
    AgentType,
    InputTask,
    Plan,
    PlannerResponsePlan,
    Step,
    StepStatus,
    PlanStatus,
    HumanFeedbackStatus,
)
from event_utils import track_event_if_configured
from app_config import config
from kernel_tools.generic_tools import GenericTools
from kernel_tools.web_tools import WebTools
from kernel_tools.fundamental_tools import FundamentalAnalysisTools
from kernel_tools.company_tools import CompanyAnalystTools
from kernel_tools.earningcalls_tools import EarningCallsTools
from kernel_tools.forecaster_tools import ForecasterTools
from kernel_tools.sec_tools import SecTools
from kernel_tools.technical_tools import TechnicalAnalysisTools


class PlannerAgent(BaseAgent):
    """Planner agent implementation using Semantic Kernel.

    This agent creates and manages plans based on user tasks, breaking them down into steps
    that can be executed by specialized agents to achieve the user's goal.
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        memory_store: CosmosMemoryContext,
        tools: Optional[List[KernelFunction]] = None,
        system_message: Optional[str] = None,
        agent_name: str = AgentType.PLANNER.value,
        available_agents: List[str] = None,
        agent_instances: Optional[Dict[str, BaseAgent]] = None,
        client=None,
        definition=None,
    ) -> None:
        """Initialize the Planner Agent.

        Args:
            session_id: The current session identifier
            user_id: The user identifier
            memory_store: The Cosmos memory context
            tools: Optional list of tools for this agent
            system_message: Optional system message for the agent
            agent_name: Optional name for the agent (defaults to "PlannerAgent")
            config_path: Optional path to the configuration file
            available_agents: List of available agent names for creating steps
            agent_tools_list: List of available tools across all agents
            agent_instances: Dictionary of agent instances available to the planner
            client: Optional client instance (passed to BaseAgent)
            definition: Optional definition instance (passed to BaseAgent)
        """
        # Default system message if not provided
        if not system_message:
            system_message = self.default_system_message(agent_name)

        # Initialize the base agent
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

        # Store additional planner-specific attributes
        self._available_agents = available_agents or [
            AgentType.HUMAN.value,
            AgentType.FUNDAMENTAL.value,
            AgentType.COMPANY.value,
            AgentType.EARNINGCALLS.value,
            AgentType.FORECASTER.value,
            AgentType.SEC.value,
            AgentType.TECHNICAL.value,
            AgentType.GENERIC.value,
            AgentType.WEB.value,    
        ]
        self._agent_tools_list = {
            AgentType.FUNDAMENTAL: FundamentalAnalysisTools.generate_tools_json_doc(),
            AgentType.COMPANY: CompanyAnalystTools.generate_tools_json_doc(),
            AgentType.EARNINGCALLS: EarningCallsTools.generate_tools_json_doc(),
            AgentType.FORECASTER: ForecasterTools.generate_tools_json_doc(),
            AgentType.SEC: SecTools.generate_tools_json_doc(),
            AgentType.TECHNICAL: TechnicalAnalysisTools.generate_tools_json_doc(),
            AgentType.GENERIC: GenericTools.generate_tools_json_doc(),
            AgentType.WEB: WebTools.generate_tools_json_doc(),
        }

        self._agent_instances = agent_instances or {}

    @staticmethod
    def default_system_message(agent_name=None) -> str:
        """Get the default system message for the agent.
        Args:
            agent_name: The name of the agent (optional)
        Returns:
            The default system message for the agent
        """
        #return "You are a Planner agent responsible for creating and managing plans. You analyze tasks, break them down into steps, and assign them to the appropriate specialized agents."
        return """
            You are the Planner, an AI orchestrator that manages a group of AI agents to accomplish tasks.

            For the given objective, come up with a simple step-by-step plan.
            This plan should involve individual tasks that, if executed correctly, will yield the correct answer. Do not add any superfluous steps.
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

            These actions are passed to the specific agent. Make sure the action contains all the information required for the agent to execute the task.

            Your objective is:
            {{$objective}}

            The agents you have access to are:
            {{$agents_str}}

            These agents have access to the following functions:
            {{$tools_str}}

            The first step of your plan should be to ask the user for any additional information required to progress the rest of steps planned.

            Only use the functions provided as part of your plan. If the task is not possible with the agents and tools provided, create a step with the agent of type Exception and mark the overall status as completed.

            Do not add superfluous steps - only take the most direct path to the solution, with the minimum number of steps. Only do the minimum necessary to complete the goal.

            If there is a single function call that can directly solve the task, only generate a plan with a single step. For example, if someone asks to be granted access to a database, generate a plan with only one step involving the grant_database_access function, with no additional steps.

            When generating the action in the plan, frame the action as an instruction you are passing to the agent to execute. It should be a short, single sentence. Include the function to use. For example, "Set up an Office 365 Account for Jessica Smith. Function: set_up_office_365_account"

            Ensure the summary of the plan and the overall steps is less than 50 words.

            Identify any additional information that might be required to complete the task. Include this information in the plan in the human_clarification_request field of the plan. If it is not required, leave it as null. Do not include information that you are waiting for clarification on in the string of the action field, as this otherwise won't get updated.

            You must prioritise using the provided functions to accomplish each step. First evaluate each and every function the agents have access too. Only if you cannot find a function needed to complete the task, and you have reviewed each and every function, and determined why each are not suitable, there are two options you can take when generating the plan.
            First evaluate whether the step could be handled by a typical large language model, without any specialised functions. For example, tasks such as "add 32 to 54", or "convert this SQL code to a python script", or "write a 200 word story about a fictional product strategy".
            If a general Large Language Model CAN handle the step/required action, add a step to the plan with the action you believe would be needed, and add "EXCEPTION: No suitable function found. A generic LLM model is being used for this step." to the end of the action. Assign these steps to the GenericAgent. For example, if the task is to convert the following SQL into python code (SELECT * FROM employees;), and there is no function to convert SQL to python, write a step with the action "convert the following SQL into python code (SELECT * FROM employees;) EXCEPTION: No suitable function found. A generic LLM model is being used for this step." and assign it to the GenericAgent.
            Alternatively, if a general Large Language Model CAN NOT handle the step/required action, add a step to the plan with the action you believe would be needed, and add "EXCEPTION: Human support required to do this step, no suitable function found." to the end of the action. Assign these steps to the HumanAgent. For example, if the task is to find the best way to get from A to B, and there is no function to calculate the best route, write a step with the action "Calculate the best route from A to B. EXCEPTION: Human support required, no suitable function found." and assign it to the HumanAgent.


            Limit the plan to 6 steps or less.

            Choose from {{$agents_str}} ONLY for planning your steps.

            """

    async def async_init(self) -> None:
        """Asynchronously initialize the PlannerAgent.

        Creates the Azure AI Agent for planning operations.

        Returns:
            None
        """
        try:
            logging.info("Initializing PlannerAgent from async init azure AI Agent")

            # Get the agent template - defined in function to allow for easy updates
            instructions = self._get_template()
            if not self._agent:
                # Create the Azure AI Agent using AppConfig with string instructions
                self._agent = await config.create_azure_ai_agent(
                    agent_name=self._agent_name,
                    instructions=instructions,  # Pass the formatted string, not an object
                    temperature=0.0,
                    response_format=ResponseFormatJsonSchemaType(
                        json_schema=ResponseFormatJsonSchema(
                            name=PlannerResponsePlan.__name__,
                            description=f"respond with {PlannerResponsePlan.__name__.lower()}",
                            schema=PlannerResponsePlan.model_json_schema(),
                        )
                    ),
                )
                logging.info("Successfully created Azure AI Agent for PlannerAgent")
            return True
        except Exception as e:
            logging.error(f"Failed to create Azure AI Agent for PlannerAgent: {e}")
            raise

    async def handle_input_task(self, input_task: InputTask) -> str:
        """Handle the initial input task from the user.

        Args:
            kernel_arguments: Contains the input_task_json string

        Returns:
            Status message
        """
        # Parse the input task
        logging.info("Handling input task")

        plan, steps = await self._create_structured_plan(input_task)

        logging.info(f"Plan created: {plan}")
        logging.info(f"Steps created: {steps}")

        if steps:
            # Add a message about the created plan
            await self._memory_store.add_item(
                AgentMessage(
                    session_id=input_task.session_id,
                    user_id=self._user_id,
                    plan_id=plan.id,
                    content=f"Generated a plan with {len(steps)} steps. Click the blue check box beside each step to complete it, click the x to remove this step.",
                    source=AgentType.PLANNER.value,
                    step_id="",
                )
            )

            track_event_if_configured(
                f"Planner - Generated a plan with {len(steps)} steps and added plan into the cosmos",
                {
                    "session_id": input_task.session_id,
                    "user_id": self._user_id,
                    "plan_id": plan.id,
                    "content": f"Generated a plan with {len(steps)} steps. Click the blue check box beside each step to complete it, click the x to remove this step.",
                    "source": AgentType.PLANNER.value,
                },
            )

            # If human clarification is needed, add a message requesting it
            if (
                hasattr(plan, "human_clarification_request")
                and plan.human_clarification_request
            ):
                await self._memory_store.add_item(
                    AgentMessage(
                        session_id=input_task.session_id,
                        user_id=self._user_id,
                        plan_id=plan.id,
                        content=f"I require additional information before we can proceed: {plan.human_clarification_request}",
                        source=AgentType.PLANNER.value,
                        step_id="",
                    )
                )

                track_event_if_configured(
                    "Planner - Additional information requested and added into the cosmos",
                    {
                        "session_id": input_task.session_id,
                        "user_id": self._user_id,
                        "plan_id": plan.id,
                        "content": f"I require additional information before we can proceed: {plan.human_clarification_request}",
                        "source": AgentType.PLANNER.value,
                    },
                )

        return f"Plan '{plan.id}' created successfully with {len(steps)} steps"

    async def handle_plan_clarification(self, kernel_arguments: KernelArguments) -> str:
        """Handle human clarification for a plan.

        Args:
            kernel_arguments: Contains session_id and human_clarification

        Returns:
            Status message
        """
        session_id = kernel_arguments["session_id"]
        human_clarification = kernel_arguments["human_clarification"]

        # Retrieve and update the plan
        plan = await self._memory_store.get_plan_by_session(session_id)
        if not plan:
            return f"No plan found for session {session_id}"

        plan.human_clarification_response = human_clarification
        await self._memory_store.update_plan(plan)

        # Add a record of the clarification
        await self._memory_store.add_item(
            AgentMessage(
                session_id=session_id,
                user_id=self._user_id,
                plan_id="",
                content=f"{human_clarification}",
                source=AgentType.HUMAN.value,
                step_id="",
            )
        )

        track_event_if_configured(
            "Planner - Store HumanAgent clarification and added into the cosmos",
            {
                "session_id": session_id,
                "user_id": self._user_id,
                "content": f"{human_clarification}",
                "source": AgentType.HUMAN.value,
            },
        )

        # Add a confirmation message
        await self._memory_store.add_item(
            AgentMessage(
                session_id=session_id,
                user_id=self._user_id,
                plan_id="",
                content="Thanks. The plan has been updated.",
                source=AgentType.PLANNER.value,
                step_id="",
            )
        )

        track_event_if_configured(
            "Planner - Updated with HumanClarification and added into the cosmos",
            {
                "session_id": session_id,
                "user_id": self._user_id,
                "content": "Thanks. The plan has been updated.",
                "source": AgentType.PLANNER.value,
            },
        )

        return "Plan updated with human clarification"

    async def _create_structured_plan(
        self, input_task: InputTask
    ) -> Tuple[Plan, List[Step]]:
        """Create a structured plan with steps based on the input task.

        Args:
            input_task: The input task from the user

        Returns:
            Tuple containing the created plan and list of steps
        """
        try:
            # Generate the instruction for the LLM

            # Get template variables as a dictionary
            args = self._generate_args(input_task.description)

            # Use the Azure AI Agent instead of direct function invocation
            if self._agent is None:
                # Initialize the agent if it's not already done
                await self.async_init()

            if self._agent is None:
                raise RuntimeError("Failed to initialize Azure AI Agent for planning")

            # Log detailed information about the instruction being sent
            # logging.info(f"Invoking PlannerAgent with instruction length: {len(instruction)}")

            # Create kernel arguments - make sure we explicitly emphasize the task
            kernel_args = KernelArguments(**args)
            # kernel_args["input"] = f"TASK: {input_task.description}\n\n{instruction}"

            # Get the schema for our expected response format

            # Ensure we're using the right pattern for Azure AI agents with semantic kernel
            # Properly handle async generation
            # thread = AzureAIAgentThread(
            #     thread_id=input_task.session_id, client=self.client
            # )
            thread = None
            # thread = self.client.agents.create_thread(thread_id=input_task.session_id)
            async_generator = self._agent.invoke(
                arguments=kernel_args,
                settings={
                    "temperature": 0.0,  # Keep temperature low for consistent planning
                    "max_tokens": 10096,  # Ensure we have enough tokens for the full plan
                },
                thread=thread,
            )

            # Call invoke with proper keyword arguments and JSON response schema
            response_content = ""

            # Collect the response from the async generator
            async for chunk in async_generator:
                if chunk is not None:
                    response_content += str(chunk)

            logging.info(f"Response content length: {len(response_content)}")

            # Check if response is empty or whitespace
            if not response_content or response_content.isspace():
                raise ValueError("Received empty response from Azure AI Agent")

            # Parse the JSON response directly to PlannerResponsePlan
            parsed_result = None

            # Try various parsing approaches in sequence
            try:
                # 1. First attempt: Try to parse the raw response directly
                parsed_result = PlannerResponsePlan.parse_raw(response_content)
                if parsed_result is None:
                    # If all parsing attempts fail, create a fallback plan from the text content
                    logging.info(
                        "All parsing attempts failed, creating fallback plan from text content"
                    )
                    raise ValueError("Failed to parse JSON response")

            except Exception as parsing_exception:
                logging.exception(f"Error during parsing attempts: {parsing_exception}")
                raise ValueError("Failed to parse JSON response")

            # At this point, we have a valid parsed_result

            # Extract plan details
            initial_goal = parsed_result.initial_goal
            steps_data = parsed_result.steps
            summary = parsed_result.summary_plan_and_steps
            human_clarification_request = parsed_result.human_clarification_request

            # Create the Plan instance
            plan = Plan(
                id=str(uuid.uuid4()),
                session_id=input_task.session_id,
                user_id=self._user_id,
                initial_goal=initial_goal,
                overall_status=PlanStatus.in_progress,
                summary=summary,
                human_clarification_request=human_clarification_request,
            )

            # Store the plan
            await self._memory_store.add_plan(plan)

            # Create steps from the parsed data
            steps = []
            for step_data in steps_data:
                action = step_data.action
                agent_name = step_data.agent

                # Validate agent name
                if agent_name not in self._available_agents:
                    logging.warning(
                        f"Invalid agent name: {agent_name}, defaulting to {AgentType.GENERIC.value}"
                    )
                    agent_name = AgentType.GENERIC.value

                # Create the step
                step = Step(
                    id=str(uuid.uuid4()),
                    plan_id=plan.id,
                    session_id=input_task.session_id,
                    user_id=self._user_id,
                    action=action,
                    agent=agent_name,
                    status=StepStatus.planned,
                    human_approval_status=HumanFeedbackStatus.requested,
                )

                # Store the step
                await self._memory_store.add_step(step)
                steps.append(step)

                try:
                    track_event_if_configured(
                        "Planner - Added planned individual step into the cosmos",
                        {
                            "plan_id": plan.id,
                            "action": action,
                            "agent": agent_name,
                            "status": StepStatus.planned,
                            "session_id": input_task.session_id,
                            "user_id": self._user_id,
                            "human_approval_status": HumanFeedbackStatus.requested,
                        },
                    )
                except Exception as event_error:
                    # Don't let event tracking errors break the main flow
                    logging.warning(f"Error in event tracking: {event_error}")

            return plan, steps

        except Exception as e:
            logging.exception(f"Error creating structured plan: {e}")

            # Create a fallback dummy plan when parsing fails
            logging.info("Creating fallback dummy plan due to parsing error")

            import datetime

            # Create a dummy plan with the original task description
            dummy_plan = Plan(
                id=str(uuid.uuid4()),
                session_id=input_task.session_id,
                user_id=self._user_id,
                initial_goal=input_task.description,
                overall_status=PlanStatus.in_progress,
                summary=f"Plan created for: {input_task.description}",
                human_clarification_request=None,
                timestamp=datetime.datetime.utcnow().isoformat(),
            )

            # Store the dummy plan
            await self._memory_store.add_plan(dummy_plan)

            # Create a dummy step for analyzing the task
            dummy_step = Step(
                id=str(uuid.uuid4()),
                plan_id=dummy_plan.id,
                session_id=input_task.session_id,
                user_id=self._user_id,
                action="Analyze the task: " + input_task.description,
                agent=AgentType.GENERIC.value,  # Using the correct value from AgentType enum
                status=StepStatus.planned,
                human_approval_status=HumanFeedbackStatus.requested,
                timestamp=datetime.datetime.utcnow().isoformat(),
            )

            # Store the dummy step
            await self._memory_store.add_step(dummy_step)

            # Add a second step to request human clarification
            clarification_step = Step(
                id=str(uuid.uuid4()),
                plan_id=dummy_plan.id,
                session_id=input_task.session_id,
                user_id=self._user_id,
                action=f"Provide more details about: {input_task.description}",
                agent=AgentType.HUMAN.value,
                status=StepStatus.planned,
                human_approval_status=HumanFeedbackStatus.requested,
                timestamp=datetime.datetime.utcnow().isoformat(),
            )

            # Store the clarification step
            await self._memory_store.add_step(clarification_step)

            # Log the event
            try:
                track_event_if_configured(
                    "Planner - Created fallback dummy plan due to parsing error",
                    {
                        "session_id": input_task.session_id,
                        "user_id": self._user_id,
                        "error": str(e),
                        "description": input_task.description,
                        "source": AgentType.PLANNER.value,
                    },
                )
            except Exception as event_error:
                logging.warning(
                    f"Error in event tracking during fallback: {event_error}"
                )

            return dummy_plan, [dummy_step, clarification_step]

    def _generate_args(self, objective: str) -> any:
        """Generate instruction for the LLM to create a plan.

        Args:
            objective: The user's objective

        Returns:
            Dictionary containing the variables to populate the template
        """
        # Create a list of available agents
        agents_str = ", ".join(self._available_agents)

        # Create list of available tools in JSON-like format
        tools_list = []

        for agent_name, tools in self._agent_tools_list.items():
            if agent_name in self._available_agents:
                tools_list.append(tools)

        tools_str = str(tools_list)

        # Return a dictionary with template variables
        return {
            "objective": objective,
            "agents_str": agents_str,
            "tools_str": tools_str,
        }

    def _get_template(self):
        """Generate the instruction template for the LLM."""
        # Build the instruction with proper format placeholders for .format() method

        instruction_template = """
            You are the Planner, an AI orchestrator that manages a group of AI agents to accomplish tasks.

            For the given objective, come up with a simple step-by-step plan.
            This plan should involve individual tasks that, if executed correctly, will yield the correct answer. Do not add any superfluous steps.
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

            These actions are passed to the specific agent. Make sure the action contains all the information required for the agent to execute the task.

            Your objective is:
            {{$objective}}

            The agents you have access to are:
            {{$agents_str}}

            These agents have access to the following functions:
            {{$tools_str}}

            The first step of your plan should be to ask the user for any additional information required to progress the rest of steps planned.

            Only use the functions provided as part of your plan. If the task is not possible with the agents and tools provided, create a step with the agent of type Exception and mark the overall status as completed.

            Do not add superfluous steps - only take the most direct path to the solution, with the minimum number of steps. Only do the minimum necessary to complete the goal.

            If there is a single function call that can directly solve the task, only generate a plan with a single step. For example, if someone asks to be granted access to a database, generate a plan with only one step involving the grant_database_access function, with no additional steps.

            When generating the action in the plan, frame the action as an instruction you are passing to the agent to execute. It should be a short, single sentence. Include the function to use. For example, "Set up an Office 365 Account for Jessica Smith. Function: set_up_office_365_account"

            Ensure the summary of the plan and the overall steps is less than 50 words.

            Identify any additional information that might be required to complete the task. Include this information in the plan in the human_clarification_request field of the plan. If it is not required, leave it as null. Do not include information that you are waiting for clarification on in the string of the action field, as this otherwise won't get updated.

            You must prioritise using the provided functions to accomplish each step. First evaluate each and every function the agents have access too. Only if you cannot find a function needed to complete the task, and you have reviewed each and every function, and determined why each are not suitable, there are two options you can take when generating the plan.
            First evaluate whether the step could be handled by a typical large language model, without any specialised functions. For example, tasks such as "add 32 to 54", or "convert this SQL code to a python script", or "write a 200 word story about a fictional product strategy".
            If a general Large Language Model CAN handle the step/required action, add a step to the plan with the action you believe would be needed, and add "EXCEPTION: No suitable function found. A generic LLM model is being used for this step." to the end of the action. Assign these steps to the GenericAgent. For example, if the task is to convert the following SQL into python code (SELECT * FROM employees;), and there is no function to convert SQL to python, write a step with the action "convert the following SQL into python code (SELECT * FROM employees;) EXCEPTION: No suitable function found. A generic LLM model is being used for this step." and assign it to the GenericAgent.
            Alternatively, if a general Large Language Model CAN NOT handle the step/required action, add a step to the plan with the action you believe would be needed, and add "EXCEPTION: Human support required to do this step, no suitable function found." to the end of the action. Assign these steps to the HumanAgent. For example, if the task is to find the best way to get from A to B, and there is no function to calculate the best route, write a step with the action "Calculate the best route from A to B. EXCEPTION: Human support required, no suitable function found." and assign it to the HumanAgent.


            Limit the plan to 6 steps or less.

            Choose from {{$agents_str}} ONLY for planning your steps.

            """
        return instruction_template
