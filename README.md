# Danish National Travel Survey RAG App

A RAG application built to answer questions about the Danish National Travel Survey (TU). 
---
The app retrieves, processes, and indexes the TU documentation, and lets users query all data found on the [TU website](https://www.man.dtu.dk/myndighedsbetjening/transportvaneundersoegelsen-tu-) quickly and intuitively. 

If the app is still active, you can launch it here: [https://transportsurveyrag.streamlit.app/](https://transportsurveyrag.streamlit.app/).

## Overview

The Danish National Travel Survey has tracked how Danish residents travel since 1975. Our RAG app consolidates web pages and PDFs from the TU website into a centralized knowledge base. Users can ask questions and receive fast, relevant responses backed by the original documentation.

This project was developed to support DTU Management that manages and communicates everything around TU with the Danish Ministry of Transport.

---

## Key Features

- **Automated Data Ingestion:**  
  A custom crawler retrieves TU-related pages from the [DTU Management Engineering sitemap](https://www.man.dtu.dk/sitemap_management_engineeringdk.xml), converts HTML/PDF content into Markdown, and splits it into manageable chunks.

- **Summaries & Embeddings:**  
  We use ChatGPT to generate concise titles and summaries for each chunk, and use OpenAI’s embedding API produces semantic vectors for similarity searches.

- **Interactive UI:**  
  The Streamlit-based interface provides a chat window where users can enter their questions and receive contextual answers alongside links to the source content.

- **Structured Agent Tools:**  
  Using the Pydantic AI framework, we've built functions callable by out AI agent to retrieve documentation, list available pages, and fetch full page content—all in an organized way.

---

## How It Works

1. **Data Collection & Processing:**  
   - **Sitemap Fetch & URL Filtering:**  
     The crawler downloads the XML sitemap and filters URLs with the TU-specific prefix.
   - **Content Retrieval:**  
     Using the Crawl4AI library, HTML and PDF documents are fetched asynchronously.
   - **Markdown Conversion & Chunking:**  
     Fetched content is converted to Markdown and split into ~5,000-character chunks, preserving natural text boundaries.
   - **Summaries & Embeddings:**  
     Each chunk is summarized and assigned a title using ChatGPT-4o-mini; embeddings are generated to enable semantic search.
   - **Database Storage:**  
     Processed chunks are stored in a Supabase table (`site_pages`) for fast, vector-based queries.

2. **Interactive Querying:**  
   The Streamlit UI prompts the user for a question. The system converts the query into an embedding, retrieves the most relevant documentation chunks, and builds an answer enhanced by GPT for a natural, conversational reply.

3. **Agent Tools:**  
   Functions decorated with Pydantic AI tools encapsulate actions such as retrieving documentation, listing available pages, or combining page content. This modular approach keeps the system maintainable and adaptable.

---

## Installation & Setup

## Installation & Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/danish-national-travel-survey-rag-app.git
   cd danish-national-travel-survey-rag-app

