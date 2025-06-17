from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew, Process
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SentientAgent:
    """Core agent orchestrator using CrewAI and AutoGen"""
    
    def __init__(
        self,
        llm_service,
        vector_store_service,
        vision_service,
        memory_window: int = 10
    ):
        self.llm_service = llm_service
        self.vector_store = vector_store_service
        self.vision_service = vision_service
        self.memory_window = memory_window
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Initialize CrewAI agents
        self.crewai_agents = self._setup_crewai_agents()
        
        # Initialize AutoGen agents
        self.autogen_agents = self._setup_autogen_agents()
    
    def _setup_crewai_agents(self) -> Dict[str, Agent]:
        """Setup CrewAI agents for different roles"""
        return {
            "perception": Agent(
                role="Perception Agent",
                goal="Process and understand sensory input (audio, vision)",
                backstory="Expert at understanding and interpreting sensory data",
                verbose=True,
                allow_delegation=True,
                llm=self.llm_service
            ),
            "reasoning": Agent(
                role="Reasoning Agent",
                goal="Analyze information and make decisions",
                backstory="Expert at logical reasoning and decision making",
                verbose=True,
                allow_delegation=True,
                llm=self.llm_service
            ),
            "memory": Agent(
                role="Memory Agent",
                goal="Manage and retrieve information from long-term memory",
                backstory="Expert at information management and retrieval",
                verbose=True,
                allow_delegation=True,
                llm=self.llm_service
            ),
            "personality": Agent(
                role="Personality Agent",
                goal="Maintain consistent personality and emotional responses",
                backstory="Expert at maintaining personality consistency",
                verbose=True,
                allow_delegation=True,
                llm=self.llm_service
            )
        }
    
    def _setup_autogen_agents(self) -> Dict[str, Any]:
        """Setup AutoGen agents for collaborative tasks"""
        # Create individual agents
        user_proxy = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={"work_dir": "workspace"},
            llm_config={"config_list": [{"model": "gpt-4"}]}
        )
        
        assistant = AssistantAgent(
            name="Assistant",
            llm_config={"config_list": [{"model": "gpt-4"}]},
            system_message="You are a helpful AI assistant."
        )
        
        # Create group chat
        groupchat = GroupChat(
            agents=[user_proxy, assistant],
            messages=[],
            max_round=50
        )
        
        # Create manager
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config={"config_list": [{"model": "gpt-4"}]}
        )
        
        return {
            "user_proxy": user_proxy,
            "assistant": assistant,
            "manager": manager
        }
    
    async def process_input(
        self,
        input_type: str,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process input through the agent system
        
        Args:
            input_type: Type of input ('text', 'audio', 'image')
            input_data: Input data
            context: Optional context information
            
        Returns:
            Dict containing processed response
        """
        # Create perception task
        perception_task = Task(
            description=f"Process {input_type} input and extract relevant information",
            agent=self.crewai_agents["perception"]
        )
        
        # Create reasoning task
        reasoning_task = Task(
            description="Analyze information and determine appropriate response",
            agent=self.crewai_agents["reasoning"]
        )
        
        # Create memory task
        memory_task = Task(
            description="Store and retrieve relevant information",
            agent=self.crewai_agents["memory"]
        )
        
        # Create personality task
        personality_task = Task(
            description="Generate response with consistent personality",
            agent=self.crewai_agents["personality"]
        )
        
        # Create and run crew
        crew = Crew(
            agents=list(self.crewai_agents.values()),
            tasks=[perception_task, reasoning_task, memory_task, personality_task],
            process=Process.sequential
        )
        
        result = crew.kickoff()
        
        # Update conversation history
        self.conversation_history.append({
            "timestamp": datetime.utcnow(),
            "input_type": input_type,
            "input_data": input_data,
            "response": result
        })
        
        # Trim history if needed
        if len(self.conversation_history) > self.memory_window:
            self.conversation_history = self.conversation_history[-self.memory_window:]
        
        return {
            "response": result,
            "context": context,
            "timestamp": datetime.utcnow()
        }
    
    async def collaborative_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a collaborative task using AutoGen
        
        Args:
            task_description: Description of the task
            context: Optional context information
            
        Returns:
            Dict containing task results
        """
        # Initialize chat
        chat_init = {
            "role": "user",
            "content": task_description
        }
        
        # Run collaborative task
        result = await self.autogen_agents["manager"].run(chat_init)
        
        return {
            "result": result,
            "context": context,
            "timestamp": datetime.utcnow()
        }
    
    async def get_memory_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memory context
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of relevant memory entries
        """
        # TODO: Implement vector search in memory
        return [] 