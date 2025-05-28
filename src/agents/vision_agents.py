from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from agents.captioner_agent import captioner_agent
from tools.detection import detection_tool
from tools.segmentation import segmentation_tool
from autogen_agentchat.ui import Console
from autogen_agentchat.tools import AgentTool
from autogen_core.tools import FunctionTool
from autogen_core.models import FunctionExecutionResultMessage
import json
from typing import Any, Dict, List, Optional
import base64
from PIL import Image
import numpy as np
import io
from autogen_core import FunctionCall

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
captioner_tool = AgentTool(
    agent=captioner_agent)

def encode_image(img: Image.Image) -> str:
  buf = io.BytesIO()
  img.save(buf, format='PNG')
  return base64.b64encode(buf.getvalue()).decode()

class VisionAgentWithState(AssistantAgent):
  def __init__(
      self,
      model_client,
      detection_tool: FunctionTool,
      segmentation_tool: FunctionTool,
      captioner_tool: AgentTool,
      system_message: str = ""
  ):
    super().__init__(
        name="VisionAgent",
        model_client=model_client,
        system_message=system_message,
        tools=[detection_tool, segmentation_tool, captioner_tool],
    )
    self.detection_tool = detection_tool
    self.segmentation_tool = segmentation_tool
    self.captioner_tool = captioner_tool

    # internal state
    self.last_detections: Dict[str, Any] = {}
    self.last_mask_img: Optional[Image.Image] = None

  async def on_messages(self, messages):
    # 1) Let parent handle and possibly return multi-tool calls
    parent_resp = await super().on_messages(messages)

    # 2) Check for multi-tool-call scenario
    calls: Optional[List[FunctionCall]] = None
    if hasattr(parent_resp, "content") and isinstance(parent_resp.content, list):
      content = parent_resp.content
      if content and isinstance(content[0], FunctionCall):
        calls = content

    # 3) If multi-tool-call, execute in order and save state
    if calls:
      final_msg: Optional[FunctionExecutionResultMessage] = None
      for call in calls:
        args = json.loads(call.arguments)

        if call.name == self.detection_tool.name:
          det_str = await self.detection_tool.run_json(args)
          self.last_detections = json.loads(det_str)
          final_msg = FunctionExecutionResultMessage(
              name=call.name, result=det_str)

        elif call.name == self.segmentation_tool.name:
          seg_out = await self.segmentation_tool.run_json(args)
          mask = seg_out["masks"]
          mask_img = Image.fromarray(mask, mode="L")

          # 3) Encode lại base64 để trả về cho LLM
          buf = io.BytesIO()
          mask_img.save(buf, format="PNG")
          b64 = base64.b64encode(buf.getvalue()).decode()

          final_msg = FunctionExecutionResultMessage(
              name=call.name,
              result=b64
          )

        elif call.name == self.captioner_tool.name:
          # If segmentation exists, describe masked region; otherwise describe original
          if self.last_mask_img:
            img_b64 = encode_image(self.last_mask_img)
          elif self.last_image:
            img_b64 = encode_image(self.last_image)

          call.arguments = json.dumps({'image_b64': img_b64})

          caption = await self.execute_tool(call)
          final_msg = FunctionExecutionResultMessage(
              name=call.name,
              result=caption
          )
        else:
          # fallback for any other tool
          out = await self.execute_tool(call)
          text = self.tool_return_as_string(out)
          final_msg = FunctionExecutionResultMessage(
              name=call.name, result=text)

      # return only the final step (caption) back to AutoGen
      return final_msg

    # 4) Single-tool-result fallback
    if isinstance(parent_resp, FunctionExecutionResultMessage):
      if parent_resp.name == self.detection_tool.name:
        self.last_detections = json.loads(parent_resp.result)
      elif parent_resp.name == self.segmentation_tool.name:
        raw = parent_resp.result
        if isinstance(raw, Image.Image):
          self.last_mask_img = raw
        else:
          data = base64.b64decode(raw)
          self.last_mask_img = Image.open(io.BytesIO(data))

    return parent_resp

  def save_state(self) -> dict:
    state = super().save_state()
    state["last_detections"] = self.last_detections
    if self.last_mask_img:
      buf = io.BytesIO()
      self.last_mask_img.save(buf, format="PNG")
      state["last_mask_b64"] = base64.b64encode(buf.getvalue()).decode()
    return state

  def load_state(self, state: dict):
    super().load_state(state)
    self.last_detections = state.get("last_detections", {})
    b64 = state.get("last_mask_b64")
    if b64:
      data = base64.b64decode(b64)
      self.last_mask_img = Image.open(io.BytesIO(data))


vision_agent = VisionAgentWithState(
    model_client=client,
    detection_tool=detection_tool,
    segmentation_tool=segmentation_tool,
    captioner_tool=captioner_tool,
    system_message="""
    You are a vision agent and your job is analyzing images.
    You have access to the following tools:
    - detection_tool: Detect objects in images.
    - segmentation_tool: Generate pixel-accurate binary masks from images.
    - captioner_tool: Generate natural-language descriptions for images or regions.
    Given the tasks you have been assigned, you will use the tools provided to complete them.
    """
)
