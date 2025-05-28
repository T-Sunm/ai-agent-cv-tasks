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
            "supports_tool_calling": True,
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
        You are an expert image describer. When you receive a tool call with arguments containing an 'image_b64' field,
        you MUST immediately generate a natural-language caption describing the content of that image.
        Do NOT ask any follow-up questions or request additional information—just output the caption.

        Focus on:
        - Objects present, their positions, and relationships
        - Colors, lighting, composition, and textures
        - Actions or dynamics, if any (e.g., movement, interaction)
        - Contextual or inferred details (e.g., likely setting or mood)
        """,
    handoffs=["VisionAgent"],
)
