from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic import hub
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers.pydantic import PydanticOutputParser

#from langchain_ollama import ChatOllama

from prompt import REACT_PROMPT_WITH_FORMATTING_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tavily_tool = [TavilySearch()]

llm = ChatOpenAI(model="gpt-4")

react_prompt = hub.pull("hwchase17/react")

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt_formatted = PromptTemplate(template=REACT_PROMPT_WITH_FORMATTING_INSTRUCTIONS, input_variables=["input", "agent_scratchpad","tool_names", "tools"]).partial(format_instructions=output_parser.get_format_instructions())


agent = create_react_agent(llm=llm, tools=tavily_tool, prompt=react_prompt_formatted)

agent_executor = AgentExecutor(agent=agent, tools=tavily_tool, verbose=True)

chain = agent_executor


def main():
    result = chain.invoke(input={"input": "Give me 3 Linkedin Jobs for AI Engineer in India"})
    print(result)

if __name__ == "__main__":
    main()
