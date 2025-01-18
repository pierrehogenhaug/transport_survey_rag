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

# --- Crawl4AI Imports ---
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# --- LLM + DB Imports (Example: OpenAI + Supabase) ---
# Note: These are imported for reference.
#from openai import AsyncOpenAI
#from supabase import create_client, Client

# Load .env variables
load_dotenv()

# Set a global DEBUG flag
DEBUG = True

# Normally, you would initialize your clients here.
# For this version, we'll not instantiate any external clients.
#openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#supabase: Client = create_client(
#    os.getenv("SUPABASE_URL"),
#    os.getenv("SUPABASE_SERVICE_KEY")
#)

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

        # Try to use paragraph boundaries.
        if '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break

        # Try to use sentence boundaries.
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
    Instead of making an external call to an LLM, this dummy version returns
    placeholder data and prints what would have been sent.
    """
    system_prompt = (
        "You are an AI that extracts titles and summaries from documentation chunks. "
        "Return a JSON object with 'title' and 'summary' keys. For the title, if this is the "
        "start of a doc, extract the doc's name. If middle, derive a descriptive title. For the summary, "
        "provide a concise summary of the chunk's main ideas. Keep both short but informative."
    )

    # Debug print: what would be sent to the LLM.
    if DEBUG:
        print("\n[DEBUG get_title_and_summary]")
        print(f"  URL: {url}")
        print(f"  Chunk (truncated to 1000 chars) being sent to LLM:\n  {chunk[:1000]!r}\n")
        print(f"  System prompt:\n  {system_prompt}\n")

    # Return dummy data
    dummy_response = {"title": "Dummy Title", "summary": "This is a dummy summary."}
    return dummy_response


async def get_embedding(text: str) -> List[float]:
    """
    Instead of calling the OpenAI embedding API, this dummy version returns
    a placeholder list and prints out debug information.
    """
    if DEBUG:
        # print("[DEBUG get_embedding]")
        # preview_len = 300  # Adjust as needed
        # text_preview = (text[:preview_len] + "...") if len(text) > preview_len else text
        # print(f"  Text length: {len(text)}")
        # print(f"  Text preview:\n{text_preview}\n")
        pass
    # Return a dummy embedding (e.g., vector of zeros of length 1536)
    dummy_embedding = [0.0] * 1536
    return dummy_embedding


async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """
    Process a single chunk: get a title & summary (dummy) and embedding (dummy),
    build metadata, and provide debug prints.
    """
    if DEBUG:
        print("\n[DEBUG process_chunk]")
        print(f"  URL: {url}")
        print(f"  Chunk number: {chunk_number}")
        print(f"  Chunk content (up to 500 chars):\n{chunk[:500]!r}")
        print(f"  Full chunk length: {len(chunk)}")

    extracted = await get_title_and_summary(chunk, url)
    embedding = await get_embedding(chunk)

    metadata = {
        "source": "transportvaneundersoegelsen-tu",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }

    if DEBUG:
        print("[DEBUG process_chunk result]")
        print(f"  Title: {extracted['title']}")
        print(f"  Summary: {extracted['summary']}")
        print(f"  Metadata: {metadata}\n")

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
    Instead of inserting into the database, this dummy version simply prints out
    the data that would have been inserted.
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

    if DEBUG:
        # Create a copy of data for debug printing and remove the embedding vector.
        debug_data = data.copy()
        debug_data["embedding"] = "<embedding omitted>"  # Commented out the raw embedding printing
        print("[DEBUG insert_chunk]")
        print(f"  Data to be inserted (dummy, not really inserting):\n{json.dumps(debug_data, indent=2)}\n")
    print(f"[OK] Dummy inserted chunk {chunk.chunk_number} for {chunk.url}")
    return data


async def process_and_store_document(url: str, markdown: str):
    """
    Splits the raw markdown into chunks, processes each chunk concurrently,
    and then (dummy) stores them.
    """
    if DEBUG:
        print("\n[DEBUG process_and_store_document]")
        print(f"  URL: {url}")
        print(f"  Raw markdown (up to 500 chars):\n{markdown[:500]!r}")
        print(f"  Full markdown length: {len(markdown)}\n")

    chunks = chunk_text(markdown)

    if DEBUG:
        print(f"  Number of chunks created: {len(chunks)}")

    tasks = [
        process_chunk(chunk, i, url)
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    insert_tasks = [
        insert_chunk(chunk)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)


async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """
    Crawl multiple URLs in parallel, with a concurrency limit. Each successful crawl
    follows the text => chunk => (dummy) store pipeline.
    """
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/104.0.0.0 Safari/537.36"
        ),
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
    )

    # Configure caching
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        css_selector="#main-content"
        # excluded_tags=["header", "footer", "nav", "aside", "script", "style"],
        # js_code="document.querySelectorAll('.h-top-bar').forEach(el => el.remove());"
    )

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    async def process_url(url: str):
        async with asyncio.Semaphore(max_concurrent):
            result = await crawler.arun(url=url, config=crawl_config)
            if result.success:
                print(f"[OK] Crawled => {url}")
                # Extract raw markdown. Some pages may have markdown_v2.
                markdown_raw = result.markdown_v2.raw_markdown if hasattr(result, "markdown_v2") and result.markdown_v2 else result.markdown
                if not markdown_raw:
                    print(f"  -> No markdown extracted from {url} (maybe empty).")
                    return
                await process_and_store_document(url, markdown_raw)
            else:
                print(f"[FAIL] {url} => {result.error_message}")

    try:
        await asyncio.gather(*[process_url(u) for u in urls])
    finally:
        await crawler.close()


def get_tu_urls_from_sitemap() -> List[str]:
    """
    Downloads the sitemap XML, extracts all <loc> text, and returns those that start with 
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

        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        all_urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        filtered = [u for u in all_urls if u.startswith(prefix)]
        print(f"Found {len(filtered)} matching 'transportvaneundersoegelsen-tu-' URLs.")
        return filtered

    except Exception as e:
        print(f"[ERROR fetching sitemap]: {e}")
        return []


async def main():
    # Gather the URLs from the sitemap.
    tu_urls = get_tu_urls_from_sitemap()
    if not tu_urls:
        print("[INFO] No 'transportvaneundersoegelsen-tu-' URLs found. Exiting.")
        return

    # Crawl the URLs and run the full pipeline (all dummy calls)
    await crawl_parallel(tu_urls, max_concurrent=5)


if __name__ == "__main__":
    asyncio.run(main())