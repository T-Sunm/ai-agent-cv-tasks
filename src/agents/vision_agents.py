from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_agentchat.teams import Swarm
from agents.captioner_agent import captioner_agent
from tools.detection import detection_tool
from tools.segmentation import segmentation_tool

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
    After completing your task , respond to the supervisor directly
    """,
    tools=[detection_tool, segmentation_tool],
    # Specify that this agent hands off to the captioner agent
    handoffs=["CaptionerAgent"],
)

vision_team = Swarm(
    name="VisionTeam",
    agents=[vision_agent, captioner_agent],
    description="Team giải quyết các tác vụ thị giác máy tính như captioning, detection, segmentation."
)
