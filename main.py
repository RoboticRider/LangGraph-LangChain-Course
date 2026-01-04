from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os


load_dotenv()


def main():
    print("Hello from Information Summarizer Agent")

    name = "Elon Reeve Musk"

    summary_template="""Given the following {name}, generate a concise summary highlighting their key achievements and roles. Give the Output in Points, max 3 points."""

    summary_prompt_template=PromptTemplate(input_variables=["name"] ,template=summary_template)

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    chain = summary_prompt_template | llm
    response = chain.invoke({"name" : name})

    print(response.content)


if __name__ == "__main__":
    main()
