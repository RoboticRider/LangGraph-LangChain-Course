from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

#Initializing the Tavily Client
#tavily = TavilyClient()

#Initializing the Tavily Search Tool
tavily_tool = TavilySearch()

#Creating the Tool that the LLM can call
# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches the Internet

#     Args:
#         query: The query to search for

#     Returns:
#         The search results
#     """

#     print(f"Searching for : {query}")
#     return tavily.search(query=query)

class SourceURL(BaseModel):
    """Schema for a Source URL used by the Agent"""

    url:str = Field(description="The URL of the Source") 

class AgentResponse(BaseModel):
    """Schema for the Agent Response with Answers and Sources"""

    answer:str = Field(description="The Answer to the Query")
    sources:List[SourceURL]= Field(default_factory=list, description="List of Sources used to answer the Query")   

#Creating the LLM instance
llm = ChatOpenAI()

#Mapping the Tool Function to the Tool Variable
tool = [tavily_tool]

#Creating the Agent
agent = create_agent(model=llm, tools=tool, response_format=AgentResponse)



def main():
    print("This is a Langchain Agent that searches Query over the Internet")
    result = agent.invoke({"messages": [HumanMessage(content="Search the Linkedin.com for 3 langchain developers job in India")]})
    print(result)


if __name__ == "__main__":
    main()