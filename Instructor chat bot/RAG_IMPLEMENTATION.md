# RAG Implementation Summary

## Overview
This project has been rebuilt to use **Retrieval-Augmented Generation (RAG)** with a **Vector Database (ChromaDB)**.

## What Changed

### 1. New Dependencies Added
- `chromadb==0.4.22` - Vector database for storing embeddings
- `sentence-transformers==2.2.2` - Embedding model for generating vector representations
- `PyPDF2==3.0.1` - For potential PDF support (prepared for future)
- `python-docx==1.1.0` - For potential DOCX support (prepared for future)
- `numpy==1.24.3` - Required by sentence-transformers

### 2. Architecture Changes

#### Before (Simple LLM):
- User message → System prompt → Gemini API → Response

#### After (RAG):
- User message → Embedding generation → Vector search → Context retrieval → Augmented prompt → Gemini API → Response

### 3. New Components

#### Vector Database (ChromaDB)
- **Location**: `chroma_db/` folder (created automatically)
- **Collection**: `educational_knowledge_base`
- **Embedding Model**: `all-MiniLM-L6-v2` (lightweight and efficient)
- **Storage**: Persistent (data survives restarts)

#### Document Processing
- **Location**: `documents/` folder
- **Format**: `.txt` files
- **Chunking**: 512 characters per chunk with 50 character overlap
- **Metadata**: Stores source file and subject for filtering

#### RAG Retrieval
- **Top-K**: Retrieves top 3 most relevant chunks
- **Subject Filtering**: Filters by subject when mode is specific (physics, mathematics, etc.)
- **Semantic Search**: Uses cosine similarity for finding relevant context

### 4. Key Functions Added

#### `chunk_text(text, chunk_size, overlap)`
Splits text into chunks with overlap, trying to break at sentence boundaries.

#### `load_documents_from_folder(folder_path)`
Loads all `.txt` files from the documents folder.

#### `process_and_store_documents(documents, collection)`
- Chunks documents
- Generates embeddings
- Stores in vector database

#### `retrieve_relevant_context(query, collection_obj, top_k, mode)`
- Embeds user query
- Searches vector database
- Returns top-K relevant chunks

#### `initialize_knowledge_base()`
Initializes the knowledge base on startup, loading documents if the vector DB is empty.

### 5. Modified Functions

#### `chat()` Endpoint
- Now retrieves relevant context before generating response
- Augments system instruction with retrieved context
- Returns RAG info (number of chunks retrieved) in response

### 6. New API Endpoints

#### `POST /reload-documents`
Reloads documents from the `documents/` folder into the vector database.

#### Updated `GET /health`
Now includes RAG status:
- Vector database connection status
- Number of chunks in collection
- Embedding model name

### 7. Sample Documents Created

Created sample knowledge base documents:
- `documents/physics.txt` - Physics fundamentals
- `documents/mathematics.txt` - Mathematics essentials
- `documents/chemistry.txt` - Chemistry basics
- `documents/astronomy.txt` - Astronomy fundamentals

## How It Works

1. **Initialization** (on startup):
   - Vector database is created/loaded
   - If empty, documents from `documents/` folder are loaded
   - Documents are chunked and embedded
   - Chunks are stored in ChromaDB

2. **User Query**:
   - User sends a question via `/chat` endpoint
   - Query is embedded using sentence-transformers
   - Vector database is searched for similar chunks
   - Top-K most relevant chunks are retrieved

3. **Response Generation**:
   - Retrieved context is added to the system instruction
   - Gemini API generates response using augmented context
   - Response includes RAG info (number of chunks used)

## Benefits of RAG

1. **Accuracy**: Responses are grounded in the knowledge base
2. **Updatability**: Easy to update knowledge by adding documents
3. **Subject Filtering**: Can filter retrieval by subject for relevant answers
4. **Semantic Search**: Finds relevant information even with different wording
5. **Context-Aware**: Retrieves only relevant context, reducing noise

## Next Steps

To use the RAG system:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your documents**:
   - Place `.txt` files in `documents/` folder
   - Documents are automatically loaded on first run

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Reload documents** (if you add new ones):
   - Use `POST /reload-documents` endpoint
   - Or restart the application (it will reload if vector DB is empty)

## Notes

- First run will be slower due to embedding generation
- Vector database persists, so documents don't need to be reloaded each time
- Embeddings use CPU by default (GPU acceleration possible with CUDA)
- The embedding model (`all-MiniLM-L6-v2`) is small and fast (~80MB)

