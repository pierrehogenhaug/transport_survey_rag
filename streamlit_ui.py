from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import create_client
from openai import AsyncOpenAI

# Import all the message part classes
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
from rag_agent import pydantic_ai_expert, PydanticAIDeps

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Let the user input an OpenAI API key via the sidebar.
# If provided, it overrides the value from .env.
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if not api_key:
    st.error("OpenAI API key is required to run the app.")
    st.stop()  # Halt execution until the key is provided

# Set the API key in the environment so that downstream modules (like rag_agent) can pick it up
os.environ["OPENAI_API_KEY"] = api_key

# Initialize the OpenAI client with the determined API key.
openai_client = AsyncOpenAI(api_key=api_key)

# Initialize the Supabase client.
# Note: If you're using the newer supabase-py package, you typically use create_client.
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    st.warning("Supabase URL or Service Key not found. Make sure they are set in your .env file.")
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Configure logfire to suppress warnings (optional)
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
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)

async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    try:
        # Prepare dependencies using the (now valid) API key.
        deps = PydanticAIDeps(
            supabase=supabase,
            openai_client=openai_client
        )

        # Run the agent in a stream
        async with pydantic_ai_expert.run_stream(
            user_input,
            deps=deps,
            message_history=st.session_state.messages[:-1],  # pass entire conversation so far
        ) as result:
            # We'll gather partial text to show incrementally
            partial_text = ""
            message_placeholder = st.empty()

            # Render partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                partial_text += chunk
                message_placeholder.markdown(partial_text)

            # Now that the stream is finished, we have a final result.
            # Add new messages from this run, excluding user-prompt messages
            filtered_messages = [
                msg for msg in result.new_messages()
                if not (hasattr(msg, 'parts') and any(part.part_kind == 'user-prompt' for part in msg.parts))
            ]
            st.session_state.messages.extend(filtered_messages)

            # Add the final response to the messages
            st.session_state.messages.append(
                ModelResponse(parts=[TextPart(content=partial_text)])
            )
    except Exception as e:
        st.error("An error occurred while processing your request. Please check your API key and try again.")
        # Optionally, log or print(e) for debugging.

async def main():
    st.title("Danish National Travel Survey RAG")
    st.write("Ask any questions about the Danish National Travel Survey, including data procedures, methodologies, or survey results.")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far.
    for msg in st.session_state.messages:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("Which questions do you have about TU?")

    if user_input:
        # Append a new request to the conversation
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display the user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    asyncio.run(main())