from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client
# If you're using the standard Supabase Python client:
# from supabase import create_client

# If you have your own async OpenAI client or a custom wrapper, import it here.
# Otherwise, for normal usage with the 'openai' library, you'd do:
# import openai
from openai import AsyncOpenAI

# Import your own pydantic_ai classes; adjust paths/modules as needed
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

# Load environment variables from .env (optional, if you want fallback .env values)
from dotenv import load_dotenv
load_dotenv()

# Let the user input an OpenAI API key via the sidebar
# If provided, it overrides any .env or environment-based key
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if not api_key:
    st.error("OpenAI API key is required to run the app.")
    st.stop()  # Halt execution until the key is provided

# Optionally set it in the environment as well (some libs may read from env)
os.environ["OPENAI_API_KEY"] = api_key
from rag_agent import pydantic_ai_expert, PydanticAIDeps

# Initialize OpenAI client with the provided API key
openai_client = AsyncOpenAI(api_key=api_key)

# Initialize Supabase client (this snippet is an example; adapt to your usage)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

# If using the standard Supabase Python client:
# supabase: Client = create_client(supabase_url, supabase_key)

# If you have a direct `Client` constructor:
supabase: Client = Client(
    supabase_url,
    supabase_key
)

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
        # Prepare dependencies using the user-provided API key
        deps = PydanticAIDeps(
            supabase=supabase,
            openai_client=openai_client  # pass your openai client here
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
        # Optionally log the exception 'e' here

async def main():
    st.title("Danish National Travel Survey RAG")
    st.write("Ask any questions about the Danish National Travel Survey, including data procedures, methodologies, or survey results.")

    # Initialize chat history in session state if not present
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
        # Append a new user request to the conversation
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