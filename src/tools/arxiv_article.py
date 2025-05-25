from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from typing_extensions import Annotated
from autogen_core.tools import FunctionTool

def arxiv_search(topic: Annotated[str, "Topic or query to search for papers on Arxiv"]) -> str:
  """Tool: Search for academic papers on a given topic using Arxiv."""
  arxiv_api = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
  return ArxivQueryRun(api_wrapper=arxiv_api).run(topic)


arxiv_search_tool = FunctionTool(
    arxiv_search,
    description="Search for academic papers on a given topic using Arxiv."
)
