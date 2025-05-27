from autogen_agentchat.teams import SelectorGroupChat
from agents.planning_agent import planning_agent
from agents.researcher_agent import research_agent
from agents.vision_agents import vision_agent
from agents.captioner_agent import captioner_agent
from autogen_core.models import ChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import MultiModalMessage
from PIL import Image as PILImage
from autogen_core import Image as AGImage
from pathlib import Path
import asyncio
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
selector_prompt = """Read the conversation history:
{history}
Available agents:
{roles}
Look for a line like 'NEXT_SPEAKER: <AgentName>' in the last message. If found, select that <AgentName>.
Otherwise, based on the last message, which agent from {participants} should speak next?
Output only the agent's name."""

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_mention_termination | max_messages_termination
team = SelectorGroupChat(
    [planning_agent, research_agent, vision_agent],
    model_client=client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,
)

async def main():
  image_path = Path(
      "../static/animals.jpg"
  )
#   pil_img = PILImage.open(image_path)
#   ag_image = AGImage(pil_img)

#   print(ag_image)
#   message = MultiModalMessage(
#       content=[
#           "Question: What breed is this cat? ,",
#           ag_image
#       ],
#       source="user"
#   )

  message = "Question: Detect all animals in the image? , image_path =" + \
      str(image_path)

  await Console(team.run_stream(task=message))

if __name__ == "__main__":
  asyncio.run(main())
