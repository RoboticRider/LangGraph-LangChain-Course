from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from crewai import Agent, Task, Crew

app = FastAPI()
security = HTTPBearer()


# 🔹 CrewAI logic (STATIC INPUT)
def main():

    user_input = "Build an invoice processing automation using UiPath that reads invoices from email, extracts data, and updates ERP system"

    researcher = Agent(
        role="Automation Architect",
        goal="Understand requirement and design automation solution",
        backstory="Expert in UiPath and intelligent automation",
        verbose=True,
        llm="ollama/qwen3.5:0.8b"
    )

    task = Task(
        description=f"""
        User Requirement:
        {user_input}

        Generate:
        - Step-by-step UiPath workflow
        - Clear structured output
        """,
        expected_output="Detailed structured automation steps",
        agent=researcher
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=True
    )

    result = crew.kickoff()

    print("\n🔥 AGENT OUTPUT:\n")
    print(result)

    return result


# ✅ STATIC API (no user input required)
@app.get("/run")
def run_agent(credentials: HTTPAuthorizationCredentials = Depends(security)):

    if credentials.credentials != "my-secret-token":
        raise HTTPException(status_code=401, detail="Unauthorized")

    result = main()

    return {"output": str(result)}