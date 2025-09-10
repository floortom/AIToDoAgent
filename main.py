from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

todoistApi = os.getenv("TODOIST_API_KEY")
geminiApi = os.getenv("GEMINI_API_KEY")

#  LLM initialization
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=geminiApi,
    temperature=0.3  # model's creativity level
)

# Prompt construction
systemPrompt = "You are a helpful assistant. You will help the user add tasks."
userInput = "What day is it today?"
prompt = ChatPromptTemplate(
    [
        ("system", systemPrompt),
        ("user", userInput)
    ]
)


chain = prompt | llm | StrOutputParser()
# print(chain)

response = chain.invoke({"input": userInput})
print(response)
