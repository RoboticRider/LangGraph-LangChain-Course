from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def multiply(x: float, y: float) -> float:
    """Multiply x times y."""
    return x * y


def main():
    print("Hello from langgraph-langchain-course!")

    tools = [
        TavilySearch(),
        multiply,
    ]

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a Weather assistant.",
    )

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What is the weather in New Delhi right now, "
                        "compare it with weather in Dubai, "
                        "give the answer in Celsius."
                    )
                }
            ]
        }
    )

    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
