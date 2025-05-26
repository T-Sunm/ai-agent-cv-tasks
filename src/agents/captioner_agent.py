from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

config = {
    "provider": "OpenAIChatCompletionClient",
    "config": {
        "model": "qwen2-vl-2b-instruct",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio",
        "model_info": {
            "name": "qwen2-vl-2b-instruct",
            "family": "openai",
            "supports_tool_calling": False,
            "supports_json_mode": True,
            "structured_output": True,
            "json_output": True,
            "function_calling": True,
            "vision": True,
            "parallel_tool_calls": False
        }
    }
}

client = ChatCompletionClient.load_component(config)

captioner_agent = AssistantAgent(
    "CaptionerAgent",
    description="An agent for generating detailed and accurate image descriptions.",
    model_client=client,
    system_message="""
    You are an expert image describer. When presented with an image, provide a detailed, accurate, and objective description of its visible content. Focus on aspects such as:
    - Objects present, their positions, and relationships
    - Colors, lighting, composition, and textures
    - Actions or dynamics, if any (e.g., people walking, water flowing)
    - Contextual or inferred information (e.g., likely setting, era, or activity)

    Avoid adding information that is not visible or cannot be reasonably inferred from the image. Do not speculate or inject personal opinion unless explicitly requested. If text appears in the image, transcribe it accurately.
    After generating the final answer, you MUST call: transfer_to_dispatcher()
    """,
    handoffs=["VisionAgent"],
)
