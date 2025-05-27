from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from agents.captioner_agent import captioner_agent
from tools.detection import detection_tool
from tools.segmentation import segmentation_tool
from autogen_agentchat.ui import Console
from autogen_agentchat.tools import AgentTool

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
captioner_tool = AgentTool(
    agent=captioner_agent)

vision_agent = AssistantAgent(
    "VisionAgent",
    description="An agent for analyzing images using detection and segmentation tools.",
    model_client=client,
    system_message="""
    You are a vision agent.
    Your job is Analyzing images.
    Tools:
        detection_tool: Detect objects in images.
        segmentation_tool: Generate pixel-accurate binary masks from images.
        captioner_agent: Generate natural-language descriptions for images or regions.

    Given the tasks you have been assigned, you will use the tools provided to analyze images.
    After completing your task and formulating your response to the supervisor, you MUST explicitly state who should speak next by ending your message with a line:
    'NEXT_SPEAKER: PlanningAgent'
    """,
    tools=[detection_tool, segmentation_tool, captioner_tool],
)
