import key_param
from pymongo import MongoClient
from langchain.agents import tool
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import voyageai

def main():
    """
    Main function to initialize and execute the graph.
    """
    
    mongodb_client, vs_collection, full_collection = init_mongodb()
    
    tools = [
        get_information_for_question_answering,
        get_page_content_for_summarization
    ]
    
    llm = ChatOpenAI(openai_api_key=key_param.openai_api_key, temperature=0, model="gpt-4o")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "You are a helpful AI assistant."
                " You are provided with tools to answer questions and summarize technical documentation related to MongoDB."
                " Think step-by-step and use these tools to get the information required to answer the user query."
                " Do not re-run tools unless absolutely necessary."
                " If you are not able to get enough information using the tools, reply with I DON'T KNOW."
                " You have access to the following tools: {tool_names}."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    
    bind_tools = llm.bind_tools(tools)
    
    llm_with_tools = prompt | bind_tools
    
    tool_call_check = llm_with_tools.invoke(["What are some best practices for data backups in MongoDB?"]).tool_calls
    print("Tool call check:")
    print(tool_call_check)