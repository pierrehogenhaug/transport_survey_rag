# Danish National Travel Survey RAG App
![](https://github.com/pierrehogenhaug/transport_survey_rag/blob/main/transport_rag_gif.gif)


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

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/danish-national-travel-survey-rag-app.git
   cd transport_survey_rag
   
2. **Create and Activate a Virtual Environment:**
   ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows

3. **Install Dependencies:**
Install the required Python packages via pip:
   ```bash
    pip install -r requirements.txt

4. **Configure Environment Variables:**
Create a .env file in the project root and add your credentials:
   ```bash
    OPENAI_API_KEY=your_openai_api_key_here
    SUPABASE_URL=your_supabase_url_here
    SUPABASE_SERVICE_KEY=your_supabase_service_key_here

5. **Run the Data Pipeline:**
The crawler script will fetch and process TU documentation, then store the chunks in Supabase.
   ```bash
    python crawler.py

6. **Launch the Interactive App:**
Start the Streamlit app with:
   ```bash
    streamlit run streamlit_ui.py
When the app loads, enter your OpenAI API key in the sidebar as prompted.

---

## Usage
Once the app is running, you can:

**Ask Questions**:
Type any query related to the Danish National Travel Survey into the chat input. The system will convert your query into an embedding, search the indexed documentation, and display an answer with relevant source excerpts.

**Explore Documentation**:
The built-in tools allow you to retrieve full documentation pages or a list of available pages for further exploration.

---

## Project Structure
`crawler.py`
Handles the retrieval, processing, and storage of TU documentation. This includes crawling the TU pages, converting content to Markdown, chunking the text, generating titles and summaries with GPT, and saving everything to Supabase.

`streamlit_ui.py`
Provides the interactive user interface using Streamlit. Users can input queries and receive contextual responses based on the processed documentation.

`rag_agent.py`
Contains the core logic for the retrieval-augmented generation (RAG) agent. It uses Pydantic AI to manage tool functions for fetching and formatting the relevant documentation.

--- 

## Deployment
**Streamlit Cloud:**
The application is hosted on Streamlit Cloud. Configure your environment variables in the secrets.toml file for a secure deployment.

**Local Deployment:**
Follow the steps above to set up your environment locally. Once configured, you can run the app with `streamlit run streamlit_ui.py` and test all functionalities on your own machine.
