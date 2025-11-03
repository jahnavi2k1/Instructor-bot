# Educational Assistant Chatbot with RAG

An AI-powered educational chatbot using **Retrieval-Augmented Generation (RAG)** and **Vector Database** technology, designed to provide accurate, context-aware answers about Physics, Mathematics, Chemistry, and Astronomy.

## Features

- 🔍 **RAG (Retrieval-Augmented Generation)**: Answers are augmented with relevant context from the knowledge base
- 🗄️ **Vector Database (ChromaDB)**: Efficient semantic search using vector embeddings
- 📚 **Knowledge Base**: Automatic document ingestion and chunking from text files
- 🎓 **Multiple Subjects**: Physics, Mathematics, Chemistry, and Astronomy modes
- 💬 **Chat History**: Maintains context across conversations
- 🎨 **Modern UI**: Beautiful, responsive web interface
- ⚡ **Semantic Search**: Finds relevant information using embeddings (sentence-transformers)

## Architecture

This project uses **RAG (Retrieval-Augmented Generation)** architecture:

1. **Document Processing**: Text documents are chunked and embedded
2. **Vector Storage**: Chunks are stored in ChromaDB with vector embeddings
3. **Retrieval**: User queries are embedded and semantically searched in the vector database
4. **Augmentation**: Retrieved context is added to the prompt
5. **Generation**: Gemini AI generates responses using the augmented context

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your Gemini API key:**
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
   
   Or update the `GEMINI_API_KEY` variable in `app.py` (for development only).

3. **Prepare knowledge base documents:**
   - Place `.txt` files in the `documents/` folder
   - Documents will be automatically loaded and chunked on first run
   - Sample documents are included for Physics, Mathematics, Chemistry, and Astronomy

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open in browser:**
   Navigate to `http://localhost:8000`

## Knowledge Base

The chatbot uses a knowledge base stored in a vector database:

- **Location**: `documents/` folder (automatically created)
- **Format**: `.txt` files
- **Processing**: Documents are automatically chunked (512 chars) with overlap (50 chars)
- **Embeddings**: Uses `all-MiniLM-L6-v2` model for generating embeddings
- **Storage**: ChromaDB persistent vector database

### Adding Documents

1. Add `.txt` files to the `documents/` folder
2. Files are automatically loaded on startup (if vector DB is empty)
3. To reload documents, use the `/reload-documents` API endpoint

### Document Naming

Document filenames determine the subject:
- `physics.txt` → Physics subject
- `mathematics.txt` → Mathematics subject
- `chemistry.txt` → Chemistry subject
- `astronomy.txt` → Astronomy subject

## Usage Examples

- "What is Newton's second law of motion?"
- "Explain the quadratic formula"
- "How do chemical bonds form?"
- "Describe the solar system"
- "What is quantum mechanics?"

## Modes

- **All Subjects**: General educational assistant (default)
- **Physics**: Specialized in physics topics
- **Mathematics**: Specialized in mathematics topics
- **Chemistry**: Specialized in chemistry topics
- **Astronomy**: Specialized in astronomy topics

Each mode filters retrieved context by subject for more relevant answers.

## API Endpoints

### Web Interface
- `GET /`: Web interface

### Chat
- `POST /chat`: Send chat message with RAG
  ```json
  {
    "message": "What is Newton's second law?",
    "history": [{"role": "user", "text": "..."}, {"role": "model", "text": "..."}],
    "mode": "physics"
  }
  ```
  Response includes:
  ```json
  {
    "reply": "Response text...",
    "mode": "physics",
    "rag_info": {
      "retrieved_chunks": 3,
      "has_context": true
    }
  }
  ```

### Knowledge Base Management
- `POST /reload-documents`: Reload documents from `documents/` folder into vector database
  ```json
  {
    "status": "success",
    "message": "Reloaded 4 documents",
    "chunks": 12
  }
  ```

### Health Check
- `GET /health`: Health check with RAG status
  ```json
  {
    "status": "healthy",
    "timestamp": "2025-01-11T10:55:00",
    "rag_status": {
      "vector_db": "connected",
      "collection_count": 12,
      "embedding_model": "all-MiniLM-L6-v2"
    }
  }
  ```

## Technologies

- **Flask**: Web framework
- **Google Gemini AI**: Language model for generation
- **ChromaDB**: Vector database for semantic search
- **sentence-transformers**: Embedding model for text vectors
- **Python**: Backend implementation

## How RAG Works

1. **Document Ingestion**: Text files are loaded and split into chunks
2. **Embedding Generation**: Each chunk is converted to a vector embedding
3. **Vector Storage**: Embeddings are stored in ChromaDB with metadata
4. **Query Processing**: User queries are embedded using the same model
5. **Semantic Search**: Top-K most similar chunks are retrieved
6. **Context Augmentation**: Retrieved chunks are added to the prompt
7. **Response Generation**: Gemini generates answers using augmented context

## License

MIT
