import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Environment variables load 
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not available in environment variable. Please check .env file"
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)


# Graph State define 
class ChatState(TypedDict): # define what data flow in whole chatbot workflow
    messages: Annotated[list[BaseMessage], add_messages]

conn = sqlite3.connect(database='chatbot.db', check_same_thread = False)

checkpointer = SqliteSaver(conn=conn)

# Graph Node Definition
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Graph Assembly
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Graph Compile
chatbot = graph.compile(checkpointer = checkpointer)


# Chat Execution Function
def start_chatbot():
    print("Chatbot Ready! Type 'exit' to quit.\n")

    thread_id = '1'
    config = {'configurable': {'thread_id': thread_id}}
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        initial_state = {"messages": [HumanMessage(content=user_input)]}
        result = chatbot.invoke(initial_state, config=config)

        # Fetch last message from the graph response
        raw_content = result["messages"][-1].content
        if isinstance(raw_content, list):
            clean_text = raw_content[0].get("text", "")
        else:
            clean_text = raw_content

        print(f"\nChatbot: {clean_text}\n")

    # print(chatbot.get_state(config=config))

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
if __name__ == "__main__":
    start_chatbot()
    