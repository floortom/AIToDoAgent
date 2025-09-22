from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from todoist_api_python.api import TodoistAPI


load_dotenv()

todoistApi = os.getenv("TODOIST_API_KEY")
geminiApi = os.getenv("GEMINI_API_KEY")

todoist = TodoistAPI(todoistApi)


# Add agent functionality
@tool
def add_task(task, desc=None):
    """
    Add a new task to the user's task list. Use this when the user wants to create a new task.
    :return:
    """
    todoist.add_task(content=task,
                     description=desc)


tools = [add_task]
#  LLM initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=geminiApi,
    temperature=0.3  # model's creativity level
)

# Prompt construction
systemPrompt = """
You are a helpful assistant. You will help the user add tasks.
You will also answer questions.
"""

prompt = ChatPromptTemplate(
    [
        ("system", systemPrompt),
        MessagesPlaceholder("history"),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ]
)


# chain = prompt | llm | StrOutputParser()
# print(chain)
agent = create_openai_tools_agent(llm, tools, prompt)
agentExec = AgentExecutor(agent=agent, tools=tools, verbose=False)

# response = chain.invoke({"input": userInput})

history = []
while True:
    userInput = input("You:")
    response = agentExec.invoke({"input": userInput,
                                 "history": history,
                                 })
    print(response["output"])
    history.append(HumanMessage(content=userInput))
    history.append(AIMessage(content=response["output"]))
