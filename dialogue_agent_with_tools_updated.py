from typing import List, Optional

from xagent import XAgent
from xagent.memory import XAgentMemory
from xagent.output_parser import XAgentOutputParser
from xagent.callbacks import XAgentCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, SystemMessage
from agents.agent_simulations.agent.dialogue_agent import DialogueAgent
from agents.conversational.output_parser import ConvoOutputParser
from config import Config
from memory.zep.zep_memory import ZepMemory
from services.run_log import RunLogsManager
from typings.agent import AgentWithConfigsOutput


class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        agent_with_configs: AgentWithConfigsOutput,
        system_message: SystemMessage,
        model: ChatOpenAI,
        tools: List[any],
        session_id: str,
        sender_name: str,
        is_memory: bool = False,
        run_logs_manager: Optional[RunLogsManager] = None,
        **tool_kwargs,
    ) -> None:
        super().__init__(name, agent_with_configs, system_message, model)
        self.tools = tools
        self.session_id = session_id
        self.sender_name = sender_name
        self.is_memory = is_memory
        self.run_logs_manager = run_logs_manager

    def send(self) -> str:
        memory: XAgentMemory

        memory = XAgentMemory(session_id=self.session_id, ...)  # Update with appropriate parameters

        memory.human_name = self.sender_name
        memory.ai_name = self.agent_with_configs.agent.name
        memory.auto_save = False

        callbacks = []

        if self.run_logs_manager:
            self.model.callbacks = [self.run_logs_manager.get_agent_callback_handler()]
            callbacks.append(self.run_logs_manager.get_agent_callback_handler())

        # Initialize XAgent instead of Langchain REACT Agent
        agent = XAgent(
            self.tools,
            self.model,
            agent_type=XAgent.Type.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            memory=memory,
            callbacks=callbacks,
            agent_kwargs={
                "system_message": self.system_message.content,
                "output_parser": XAgentOutputParser(),
            },
        )

        prompt = "\n".join(self.message_history + [self.prefix])

        res = agent.run(input=prompt)

        message = AIMessage(content=res)

        return message.content
