from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os


load_dotenv()


def main():
    print("Hello from Information Summarizer Agent")

    information = "Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since 2021; as of December 2025, Forbes estimates his net worth to be around US$754 billion."

    summary_template="""Given the following {information} about a person, generate a concise summary highlighting their key achievements and roles. Give the Output in Points."""

    summary_prompt_template=PromptTemplate(input_variables=["information"] ,template=summary_template)

    llm = ChatOpenAI(model_name="gpt-5", temperature=0)

    chain = summary_prompt_template | llm
    response = chain.invoke({"information" : information})

    print(response.content)


if __name__ == "__main__":
    main()
