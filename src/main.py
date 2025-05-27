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
from typing import Sequence, List, Union
from autogen_agentchat.teams._group_chat._base_group_chat import BaseAgentEvent, BaseChatMessage

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
selector_prompt = """
Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Only select one agent."""


def candidate_func(
    messages: Sequence[Union[BaseAgentEvent, BaseChatMessage]]
) -> List[str]:
  """
  Luôn để PlanningAgent trả lời sau user.
  Sau khi PlanningAgent plan xong:
    - nếu trong text có mention ResearcherAgent thì để ResearcherAgent
    - nếu trong text có mention VisionAgent thì để VisionAgent
    - nếu mention cả hai thì chọn cả hai (model sẽ pick 1)
  Khi Researcher & Vision đã đều lên tiếng rồi thì trả lại PlanningAgent.
  Mặc định nếu không rơi vào điều kiện nào thì đánh hàng [Planning, Researcher, Vision].
  """
  last = messages[-1]
  src = last.source

  # 1) Nếu vừa là user turn, next luôn là PlanningAgent
  if src.lower() == "user":
    return [planning_agent.name]

  # 2) Nếu vừa là PlanningAgent, look for explicit mentions
  if src == planning_agent.name:
    # Lấy text của message (nếu có .to_text(), else .content)
    text = getattr(last, "to_text", lambda: last.content)()
    candidates = []
    if research_agent.name in text:
      candidates.append(research_agent.name)
    if vision_agent.name in text:
      candidates.append(vision_agent.name)
    if candidates:
      return candidates

  # 3) Nếu cả Researcher và Vision đều đã lên tiếng ở bất kỳ turn nào
  seen = {m.source for m in messages}
  if research_agent.name in seen and vision_agent.name in seen:
    return [planning_agent.name]

  return [
      planning_agent.name
  ]


text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_mention_termination | max_messages_termination
team = SelectorGroupChat(
    [planning_agent, research_agent, vision_agent],
    model_client=client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    candidate_func=candidate_func,
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
