import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

# set the event loop policy for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# --- Crawl4AI Imports ---
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# --- LLM + DB Imports (Example: OpenAI + Supabase) ---
from openai import AsyncOpenAI
from supabase import create_client, Client

# Load .env variables
load_dotenv()

# Initialize LLM and Supabase
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# DataClass to hold processed chunk data
@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """
    Splits text into smaller chunks. Tries to respect code blocks, paragraphs, 
    or sentence boundaries so you don't cut the text in awkward places.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            # Last chunk
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]

        # try paragraph boundary
        if '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break

        # If no paragraph break, try sentence break
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move to next chunk
        start = max(start + 1, end)

    return chunks


async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """
    Sends a small portion of the chunk (first ~1000 chars) to GPT-4 (or any LLM)
    to produce an approximate title & summary. Adjust as needed.
    """
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
Return a JSON object with 'title' and 'summary' keys.
For the title: If this is the start of a doc, extract the doc's name. If middle, derive a descriptive title.
For the summary: Provide a concise summary of the chunk's main ideas.
Keep both short but informative."""

    try:
        # Adjust model or parameters to your needs
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."
                }
            ],
            # If your library or usage pattern differs, adapt accordingly
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[ERROR get_title_and_summary]: {e}")
        return {"title": "Error extracting", "summary": "Error extracting"}


async def get_embedding(text: str) -> List[float]:
    """
    Sends content to the OpenAI Embedding endpoint to get vector representation.
    """
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",  # or your chosen embedding model
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[ERROR get_embedding]: {e}")
        # Return a zero-vector fallback if there's an error
        return [0.0] * 1536


async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """
    Process a single chunk: get a title & summary from GPT, get embeddings, build metadata.
    """
    extracted = await get_title_and_summary(chunk, url)
    embedding = await get_embedding(chunk)

    metadata = {
        "source": "transportvaneundersoegelsen-tu",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted["title"],
        summary=extracted["summary"],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )


async def insert_chunk(chunk: ProcessedChunk):
    """
    Inserts a processed chunk record into your Supabase table (e.g., 'site_pages').
    Adjust table name / structure to match your schema.
    """
    data = {
        "url": chunk.url,
        "chunk_number": chunk.chunk_number,
        "title": chunk.title,
        "summary": chunk.summary,
        "content": chunk.content,
        "metadata": chunk.metadata,
        "embedding": chunk.embedding
    }

    try:
        result = supabase.table("site_pages").insert(data).execute()
        print(f"[OK] Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"[ERROR insert_chunk]: {e}")
        return None


async def process_and_store_document(url: str, markdown: str):
    """
    Splits the raw markdown into chunks, processes each chunk concurrently,
    and stores them in the DB.
    """
    chunks = chunk_text(markdown)

    # Process each chunk in parallel
    tasks = [
        process_chunk(chunk, i, url)
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    # Insert each chunk in parallel
    insert_tasks = [
        insert_chunk(chunk)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)


async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """
    Crawl multiple URLs in parallel, respecting a concurrency limit.
    Each successful crawl yields a text => chunk => store pipeline.
    """
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
    )

    # Choose how you want the crawler to handle caching
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        css_selector="#main-content, article"
        # ,excluded_tags=["header", "footer", "nav", "aside"]
)

    # Create the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    async def process_url(url: str):
        async with asyncio.Semaphore(max_concurrent):
            result = await crawler.arun(url=url, config=crawl_config)
            if result.success:
                print(f"[OK] Crawled => {url}")
                # We'll store the "raw_markdown" (or "markdown_v2.raw_markdown")
                markdown_raw = result.markdown_v2.raw_markdown if result.markdown_v2 else result.markdown
                if not markdown_raw:
                    print(f"  -> No markdown extracted from {url} (maybe empty).")
                    return
                await process_and_store_document(url, markdown_raw)
            else:
                print(f"[FAIL] {url} => {result.error_message}")

    try:
        # Kick off parallel tasks
        await asyncio.gather(*[process_url(u) for u in urls])
    finally:
        await crawler.close()


def get_tu_urls_from_sitemap() -> List[str]:
    """
    Downloads the DTU Management Engineering sitemap XML, extracts all <loc> text,
    and returns only those that start with 
    'https://www.man.dtu.dk/myndighedsbetjening/transportvaneundersoegelsen-tu-'
    """
    sitemap_url = "https://www.man.dtu.dk/sitemap_management_engineeringdk.xml"
    prefix = "https://www.man.dtu.dk/myndighedsbetjening/transportvaneundersoegelsen-tu-"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(sitemap_url, headers=headers, timeout=20)
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)

        # The namespace for a standard sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        # Extract all URLs
        all_urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]

        # Filter out any that don't start with the prefix
        filtered = [u for u in all_urls if u.startswith(prefix)]
        print(f"Found {len(filtered)} matching 'transportvaneundersoegelsen-tu-' URLs.")
        return filtered

    except Exception as e:
        print(f"[ERROR fetching sitemap]: {e}")
        return []


async def main():
    # 1) Gather all relevant URLs from the sitemap
    tu_urls = get_tu_urls_from_sitemap()
    if not tu_urls:
        print("[INFO] No 'transportvaneundersoegelsen-tu-' URLs found. Exiting.")
        return

    # 2) Crawl them in parallel and store results
    await crawl_parallel(tu_urls, max_concurrent=5)


if __name__ == "__main__":
    asyncio.run(main())