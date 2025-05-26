from autogen_agentchat.teams import SelectorGroupChat
from agents.planning_agent import planning_agent
from agents.researcher_agent import research_agent
from agents.vision_agents import vision_team
from autogen_core.models import ChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

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
selector_prompt = """Select an agent to perform task.
    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the planner agent has assigned tasks before other agents start working.
    Only select one agent.
"""

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_mention_termination | max_messages_termination
team = SelectorGroupChat(
    [planning_agent, research_agent, vision_team],
    model_client=client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    # Allow an agent to speak multiple turns in a row.
    allow_repeated_speaker=True,
)
