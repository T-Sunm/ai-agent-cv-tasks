from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from tools.arxiv_article import arxiv_search_tool
from tools.wikipedia_article import wikipedia_article_tool
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

research_agent = AssistantAgent(
    "ResearchAgent",
    description="An agent for performing research tasks.",
    model_client=client,
    system_message="""
    You are a research agent.
    Your job is to handle research tasks.
    Tools:
        wikipedia_article_tool: Search Wikipedia for relevant topics.
        arxiv_search_tool: Find academic papers on arXiv.
    Focus:
    - Retrieve accurate, concise, and relevant information
    - Prefer high-quality, reputable, and recent sources
    - Avoid speculative or non-verifiable content
    
    Use the tools as needed to complete assigned tasks.
    After completing your task , respond to the supervisor directly
    """,
    tools=[wikipedia_article_tool, arxiv_search_tool]
)
