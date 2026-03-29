from crewai import Agent, Task, Crew, LLM

def main():

    user_input = "Explain Machine Learning and summarize it"

    researcher = Agent(
        role="Automation Architect",
        goal="Understand requirement and design automation solution",
        backstory="Expert in UiPath and intelligent automation",
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

    print("\n🔥 AGENT OUTPUT:\n")
    print(result)


if __name__ == "__main__":
    main()