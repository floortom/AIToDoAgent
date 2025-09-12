from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor

load_dotenv()

todoistApi = os.getenv("TODOIST_API_KEY")
geminiApi = os.getenv("GEMINI_API_KEY")


# Add agent functionality
@tool
def add_task():
    """
    Add a new task to the user's task list. Use this when the user wants to create a new task.
    :return:
    """
    print("Adding a task")
    print("Task added")


tools = [add_task]
#  LLM initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=geminiApi,
    temperature=0.3  # model's creativity level
)

# Prompt construction
systemPrompt = "You are a helpful assistant. You will help the user add tasks."
userInput = "Add a task to buy milk"
prompt = ChatPromptTemplate(
    [
        ("system", systemPrompt),
        ("user", userInput),
        MessagesPlaceholder("agent_scratchpad")
    ]
)


# chain = prompt | llm | StrOutputParser()
# print(chain)
agent = create_openai_tools_agent(llm, tools, prompt)
agentExec = AgentExecutor(agent=agent, tools=tools, verbose=True)

# response = chain.invoke({"input": userInput})
response = agentExec.invoke({"input": userInput})
print(response)

