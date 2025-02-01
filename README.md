# Danish National Travel Survey RAG App

A RAG application built to answer questions about the Danish National Travel Survey (TU). 
---
The app retrieves, processes, and indexes the TU documentation, and lets users query all data found on the [TU website](https://www.man.dtu.dk/myndighedsbetjening/transportvaneundersoegelsen-tu-) quickly and intuitively. 

If the app is still active, launch it here: [https://transportsurveyrag.streamlit.app/](https://transportsurveyrag.streamlit.app/).

## Overview

The Danish National Travel Survey has tracked how Danish residents travel since 1975. Our RAG app consolidates web pages and PDFs from the TU website into a centralized knowledge base. Users can ask questions and receive fast, relevant responses backed by the original documentation.

This project was developed to support a DTU Management that manages TU data and communicates with the Danish Ministry of Transport.

---

## Key Features

- **Automated Data Ingestion:**  
  A custom crawler retrieves TU-related pages from the DTU Management Engineering sitemap, converts HTML/PDF content into Markdown, and splits it into manageable chunks.

- **Intelligent Summaries & Embeddings:**  
  A GPT-based process generates concise titles and summaries for each chunk, while OpenAI’s embedding API produces semantic vectors for similarity searches.

- **Scalable Architecture:**  
  With adjustable concurrency settings, the system efficiently handles both small and large sets of web pages.

- **Interactive UI:**  
  The Streamlit-based interface provides a chat-style experience where users can enter their questions and receive contextual answers alongside links to the source content.

- **Structured Agent Tools:**  
  Using the Pydantic AI framework, the app exposes functions to retrieve documentation, list available pages, and fetch full page content—all in a neatly organized way.

---

## How It Works

1. **Data Collection & Processing:**  
   - **Sitemap Fetch & URL Filtering:**  
     The crawler downloads an XML sitemap and filters URLs with the TU-specific prefix.
   - **Content Retrieval:**  
     Using the Crawl4AI library, HTML and PDF documents are fetched asynchronously.
   - **Markdown Conversion & Chunking:**  
     Fetched content is converted to Markdown and split into ~5,000-character chunks, preserving natural text boundaries.
   - **Summaries & Embeddings:**  
     Each chunk is summarized and assigned a title via a GPT-based model; embeddings are generated to enable semantic search.
   - **Database Storage:**  
     Processed chunks are stored in a Supabase table (`site_pages`) for fast, vector-based queries.

2. **Interactive Querying:**  
   The Streamlit UI prompts the user for a question. The system converts the query into an embedding, retrieves the most relevant documentation chunks, and builds an answer—often enhanced by GPT for a natural, conversational reply.

3. **Agent Tools:**  
   Functions decorated with Pydantic AI tools encapsulate actions such as retrieving documentation, listing available pages, or combining page content. This modular approach keeps the system maintainable and adaptable.

---

## Installation & Setup

1. **Clone the Repository:**
