from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import create_client, Client
from openai import AsyncOpenAI

# ---------------
# GET OPENAI KEY
# ---------------
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

openai_client = AsyncOpenAI(api_key=api_key)

# ---------------
# GET SUPABASE SECRETS
# ---------------
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_SERVICE_KEY"]
    supabase: Client = create_client(supabase_url, supabase_key)
except KeyError as err:
    st.write("Could not find Supabase credentials in secrets.")
    raise err

# Optionally set environment variables if needed
# os.environ["OPENAI_API_KEY"] = api_key

# Optionally configure logfire
logfire.configure(send_to_logfire='never')

from rag_agent import pydantic_ai_expert, PydanticAIDeps
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

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part):
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
    try:
        deps = PydanticAIDeps(
            supabase=supabase,
            openai_client=openai_client
        )

        async with pydantic_ai_expert.run_stream(
            user_input,
            deps=deps,
            message_history=st.session_state.messages[:-1],
        ) as result:
            partial_text = ""
            message_placeholder = st.empty()

            async for chunk in result.stream_text(delta=True):
                partial_text += chunk
                message_placeholder.markdown(partial_text)

            filtered_messages = [
                msg for msg in result.new_messages()
                if not (hasattr(msg, 'parts') and any(
                    part.part_kind == 'user-prompt' for part in msg.parts
                ))
            ]
            st.session_state.messages.extend(filtered_messages)

            st.session_state.messages.append(
                ModelResponse(parts=[TextPart(content=partial_text)])
            )

    except Exception as e:
        st.error("An error occurred while processing your request. Please check your API key and try again.")
        raise e

async def main():
    st.title("Danish National Travel Survey RAG")
    st.write("Ask any questions about TU (Danish National Travel Survey).")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the chat history
    for msg in st.session_state.messages:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            for part in msg.parts:
                display_message_part(part)

    user_input = st.chat_input("Which questions do you have about TU?")
    if user_input:
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    asyncio.run(main())