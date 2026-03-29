from fastapi import FastAPI
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM

app = FastAPI()


# ✅ Request Body Model (INPUT ARGUMENT)
class UserInput(BaseModel):
    input_text: str


# 🔹 Core CrewAI Logic
def run_crew(user_input: str):

    researcher = Agent(
        role="Expert Educator",
        goal="Understand requirement and give the answer to the query",
        backstory="Expert in Understanding things and queries and Summarizing them",
        verbose=True,
        llm=LLM(model="ollama/qwen3:0.6b", temperature=0)
    )

    task = Task(
        description=f"""
        Give me more information on:
        {user_input} and Summarize it as well.

        Generate:
        - Bullet points
        - Clear structured output
        """,
        expected_output="Summary generated for the given task in description",
        agent=researcher
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=True
    )

    result = crew.kickoff()
    return str(result)


# ✅ API Endpoint
@app.post("/run-agent")
def run_agent(data: UserInput):
    output = run_crew(data.input_text)
    return {
        "status": "success",
        "input": data.input_text,
        "output": output
    }