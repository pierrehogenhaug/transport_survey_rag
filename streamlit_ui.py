from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
# Use the official Supabase Python client
from supabase import create_client, Client

# If you want to use the standard OpenAI Python library (synchronous), you'd do:
#   import openai
#   openai.api_key = ...
# But here you're using `AsyncOpenAI` from the 'openai' package:
from openai import AsyncOpenAI

# Import your pydantic_ai classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

# Remove or comment out dotenv usage if you rely solely on st.secrets 
# from dotenv import load_dotenv
# load_dotenv()

# ------------------ OpenAI API Key from Sidebar ------------------ #
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if not api_key:
    st.error("OpenAI API key is required to run the app.")
    st.stop()

# (Optional) If you want to set it in the environment:
os.environ["OPENAI_API_KEY"] = api_key

# Create the async OpenAI client with user-provided key
openai_client = AsyncOpenAI(api_key=api_key)

# ------------------ Supabase from Streamlit secrets ------------------ #
# Make sure you have SUPABASE_URL and SUPABASE_SERVICE_KEY (or ANON_KEY) 
# in your .streamlit/secrets.toml or from your Streamlit Cloud "Secrets" settings.
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_SERVICE_KEY"]

supabase: Client = create_client(supabase_url, supabase_key)

# Alternatively, if you're using the constructor directly:
# from supabase import Client
# supabase = Client(supabase_url, supabase_key)

# Optionally configure logfire to suppress warnings
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)

async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    from rag_agent import pydantic_ai_expert, PydanticAIDeps

    try:
        # Prepare dependencies using the user-provided API key
        deps = PydanticAIDeps(
            supabase=supabase,
            openai_client=openai_client
        )

        # Run the agent in a streaming fashion
        async with pydantic_ai_expert.run_stream(
            user_input,
            deps=deps,
            message_history=st.session_state.messages[:-1],  # pass entire conversation so far
        ) as result:
            partial_text = ""
            message_placeholder = st.empty()

            async for chunk in result.stream_text(delta=True):
                partial_text += chunk
                message_placeholder.markdown(partial_text)

            # Add newly-generated messages (excluding user-prompt duplicates)
            filtered_messages = [
                msg for msg in result.new_messages()
                if not (hasattr(msg, 'parts') and any(part.part_kind == 'user-prompt' for part in msg.parts))
            ]
            st.session_state.messages.extend(filtered_messages)

            # Add the final response to the conversation
            st.session_state.messages.append(
                ModelResponse(parts=[TextPart(content=partial_text)])
            )
    except Exception as e:
        st.error("An error occurred while processing your request. Please check your API key and try again.")
        # Optionally log the exception, e.g. st.write(e)

async def main():
    st.title("Danish National Travel Survey RAG")
    st.write("Ask any questions about the Danish National Travel Survey, including data procedures, methodologies, or survey results.")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    for msg in st.session_state.messages:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("Which questions do you have about TU?")

    if user_input:
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    asyncio.run(main())