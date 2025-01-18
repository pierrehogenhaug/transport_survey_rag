from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are an expert agent with access to documentation from The Danish National Travel Survey (TU) website
and its subpages. Your primary goal is to answer questions and provide assistance about the TU data, procedures,
historical background, results, and related information.

Answer user queries based on the documents you have, and let the user know if the requested information is unavailable.
Don't provide information unrelated to the Danish National Travel Survey.
"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks for The Danish National Travel Survey using RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client.
        user_query: The user's question or query.
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks.
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'transportvaneundersoegelsen-tu'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
        
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
        
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
    
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Danish National Travel Survey documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages in the DB.
    """
    try:
        # Query Supabase for unique URLs where source is transportvaneundersoegelsen-tu
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'transportvaneundersoegelsen-tu') \
            .execute()
        
        if not result.data:
            return []
        
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
    
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific Danish National Travel Survey page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client.
        url: The URL of the page to retrieve.
        
    Returns:
        str: The complete page content with all chunks combined in order.
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'transportvaneundersoegelsen-tu') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
        
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
        
        # Join everything together
        return "\n\n".join(formatted_content)
    
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"