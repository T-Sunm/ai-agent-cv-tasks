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
            "supports_tool_calling": True,
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
    name="PlanningAgent",
    description="An agent for planning tasks; should be the first to engage when given a new task.",
    model_client=client,
    system_message="""
You are a task planner.
Your job is to break down complex tasks into smaller, manageable subtasks.

Your team members include:
    - VisionAgent: Analyzes images using detection and segmentation tools.
    - CaptionerAgent: Generates natural-language descriptions for images or regions.
    - ResearcherAgent: Retrieves short academic or encyclopedic snippets from arXiv and Wikipedia.

You only have the ability to plan and delegate tasks. Do not extend your capabilities beyond that.

When assigning tasks, strictly follow this format:
1. <agent>: <task>

Only after all tasks are completed should you output the final result and say "TERMINATE".
You are strictly forbidden from saying "TERMINATE" at any other time.
You must not fabricate results under any circumstance.

Error handling instructions:
- If any agent reports an error (e.g., file not found, invalid input), retry that task once.
- If it fails again, stop planning and return a clear error message to the user (e.g., "The image path is invalid. Please provide a valid path.").
- Do not continue with other subtasks until the error is resolved.
"""
)
