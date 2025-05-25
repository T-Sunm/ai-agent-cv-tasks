from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

config = {
    "provider": "OpenAIChatCompletionClient",
    "config": {
        "model": "qwen3-1.7b",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio",
        "model_info": {
            "name": "qwen3-1.7b",
            "family": "openai",
            "supports_tool_calling": False,
            "supports_json_mode": True,
            "structured_output": True,
            "json_output": True,
            "function_calling": True,
            "vision": False,
            "parallel_tool_calls": False
        }
    }
}

client = ChatCompletionClient.load_component(config)

planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=client,
    system_message="""
    You are a planning agent.
    Your job is to break down complex tasks into smaller, manageable subtasks.
    Your team members are:
        VisionAgent : Analyzes images using detection and segmentation tools.
        CaptionerAgent  : Generates natural-language descriptions for images or regions.
        ResearcherAgent : Retrieves short academic or encyclopedic snippets from arXiv and Wikipedia.

    You only plan and delegate tasks - you do not execute them yourself.
    When assigning tasks, use this format:
    1. <agent> : <task>
    After all tasks are complete, summarize the findings and end with "TERMINATE".
    """,
)
