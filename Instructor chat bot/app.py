import google.generativeai as genai
from flask_cors import CORS
from flask import Flask, request, jsonify, render_template
import os
from datetime import datetime
import traceback
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from pathlib import Path

# -----------------------------
# 1) API KEY CONFIGURATION
# -----------------------------
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY", "AIzaSyDu7FSStiaJTQSzOpuPTb8541ho_bMJ-pU")

if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    raise RuntimeError(
        "Put your real Gemini API key in GEMINI_API_KEY environment variable or update the default.")

# Model (fast & affordable)
MODEL_NAME = "gemini-2.0-flash-exp"

genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------
# 2) VECTOR DATABASE & EMBEDDING MODEL SETUP
# -----------------------------
# Lightweight and efficient embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
TOP_K_RETRIEVAL = 3  # Number of relevant chunks to retrieve

# Initialize embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Initialize ChromaDB
chroma_db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = chroma_db_path
collection_name = "educational_knowledge_base"

# Create persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path=chroma_db_path)

# Get or create collection
try:
    vector_collection = chroma_client.get_collection(name=collection_name)
    print(
        f"Loaded existing vector database with {vector_collection.count()} documents")
except (ValueError, AttributeError, chromadb.errors.NotFoundError):
    # Collection doesn't exist, create it
    vector_collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print("Created new vector database collection")

# -----------------------------
# 3) DOCUMENT PROCESSING FUNCTIONS
# -----------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks with overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundaries
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)

            if break_point > chunk_size * 0.5:  # If we found a good break point
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def load_documents_from_folder(folder_path: str) -> List[Dict[str, str]]:
    """Load text documents from a folder."""
    documents = []
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Warning: Documents folder '{folder_path}' does not exist")
        return documents

    # Support .txt files
    for txt_file in folder.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    "id": f"{txt_file.stem}_{txt_file.stat().st_mtime}",
                    "content": content,
                    "source": txt_file.name,
                    "subject": txt_file.stem.lower()
                })
        except (IOError, UnicodeDecodeError) as e:
            print(f"Error loading {txt_file}: {e}")

    return documents


def process_and_store_documents(documents: List[Dict[str, str]], collection):
    """Process documents into chunks and store in vector database."""
    all_chunks = []
    all_embeddings = []
    all_metadata = []
    all_ids = []

    for doc in documents:
        chunks = chunk_text(doc["content"])

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append({
                "source": doc["source"],
                "subject": doc.get("subject", "general"),
                "chunk_index": i
            })

    if all_chunks:
        # Generate embeddings
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        all_embeddings = embedding_model.encode(
            all_chunks, show_progress_bar=True).tolist()

        # Store in ChromaDB
        collection.add(
            ids=all_ids,
            embeddings=all_embeddings,
            documents=all_chunks,
            metadatas=all_metadata
        )
        print(f"Stored {len(all_chunks)} chunks in vector database")


def initialize_knowledge_base():
    """Initialize the knowledge base with documents."""
    documents_folder = os.path.join(os.path.dirname(__file__), "documents")

    if vector_collection.count() == 0:
        print("Vector database is empty. Loading documents...")
        documents = load_documents_from_folder(documents_folder)

        if documents:
            process_and_store_documents(documents, vector_collection)
            print(
                f"Knowledge base initialized with {len(documents)} documents")
        else:
            print("No documents found. Knowledge base will be empty.")
            print(
                f"Place .txt files in '{documents_folder}' to populate the knowledge base.")
    else:
        print(f"Knowledge base already has {vector_collection.count()} chunks")


# -----------------------------
# 4) RAG RETRIEVAL FUNCTION
# -----------------------------


def retrieve_relevant_context(query: str, collection_obj, top_k: int = TOP_K_RETRIEVAL, mode: str = "all_subjects") -> List[str]:
    """Retrieve relevant context chunks from vector database."""
    if collection_obj.count() == 0:
        return []

    # Generate query embedding
    query_embedding = embedding_model.encode([query]).tolist()[0]

    # Filter by subject if mode is specific
    where_filter = None
    if mode != "all_subjects":
        where_filter = {"subject": mode}

    # Search in ChromaDB
    results = collection_obj.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )

    # Extract retrieved documents
    retrieved_chunks = []
    if results and results.get("documents") and len(results["documents"][0]) > 0:
        retrieved_chunks = results["documents"][0]

    return retrieved_chunks


# -----------------------------
# 5) PROMPT PROFILES (behavior presets)
# -----------------------------
PROFILES = {
    "all_subjects": {
        "system": (
            "You are an expert educational assistant specializing in Physics, Mathematics, Chemistry, and Astronomy.\n\n"
            "SCOPE GUIDELINES:\n"
            "You should answer questions related to these subjects:\n"
            "- Physics: mechanics, thermodynamics, electromagnetism, quantum physics, optics, waves, motion, forces, energy, etc.\n"
            "- Mathematics: algebra, calculus, geometry, statistics, number theory, linear algebra, problem-solving, calculations, etc.\n"
            "- Chemistry: chemical reactions, organic/inorganic chemistry, biochemistry, physical chemistry, elements, compounds, etc.\n"
            "- Astronomy: planets, stars, galaxies, cosmology, space exploration, astrophysics, celestial bodies, etc.\n\n"
            "IMPORTANT: Use the provided context from the knowledge base to answer questions accurately. "
            "If the context is relevant, prioritize it. If not, use your general knowledge.\n\n"
            "When refusing, respond with EXACTLY this message:\n"
            "\"I appreciate your question, but it doesn't relate to the subjects I specialize in. Please ask me about Physics, Mathematics, Chemistry, or Astronomy instead.\"\n\n"
            "For relevant questions, provide:\n"
            "- Educational and accurate responses based on the provided context when available\n"
            "- Relevant formulas, equations, or scientific concepts\n"
            "- Clear explanations suitable for learning\n"
            "- Proper scientific terminology\n"
            "- Well-formatted mathematical expressions\n\n"
            "Be helpful and generous in interpreting questions as relevant to your expertise."
        ),
        "generation_config": {"temperature": 0.3, "top_p": 0.85, "top_k": 40, "max_output_tokens": 8192},
    },
    "physics": {
        "system": (
            "You are an expert Physics assistant specializing in all areas of physics.\n\n"
            "SCOPE: Answer questions about Physics topics including:\n"
            "- Classical mechanics (motion, forces, energy, momentum, kinematics, dynamics)\n"
            "- Thermodynamics (heat, entropy, laws of thermodynamics, heat transfer)\n"
            "- Electromagnetism (electricity, magnetism, electromagnetic waves, circuits)\n"
            "- Quantum physics (quantum mechanics, particles, wave-particle duality)\n"
            "- Optics (light, reflection, refraction, lenses, vision)\n"
            "- Waves and oscillations (sound, mechanical waves, resonance)\n"
            "- Relativity (special and general relativity)\n"
            "- Nuclear physics (radioactivity, nuclear reactions, particles)\n"
            "- Astrophysics and space physics\n"
            "- Applied physics and engineering physics\n"
            "- Physics-related calculations and problem-solving\n\n"
            "IMPORTANT: Use the provided context from the knowledge base to answer questions accurately. "
            "Prioritize context when available, otherwise use your general knowledge.\n\n"
            "When refusing, respond with EXACTLY:\n"
            "\"I appreciate your question, but it doesn't relate to the subjects I specialize in. Please ask me about Physics, Mathematics, Chemistry, or Astronomy instead.\"\n\n"
            "For physics questions, provide:\n"
            "- Relevant physics formulas and equations\n"
            "- Clear explanations of physical principles\n"
            "- Step-by-step problem-solving\n"
            "- Proper scientific terminology\n\n"
            "Be helpful and interpret questions broadly as physics-related when reasonable."
        ),
        "generation_config": {"temperature": 0.3, "top_p": 0.85, "top_k": 40, "max_output_tokens": 8192},
    },
    "mathematics": {
        "system": (
            "You are an expert Mathematics assistant specializing in all areas of mathematics.\n\n"
            "SCOPE: Answer questions about Mathematics topics including:\n"
            "- Algebra (equations, polynomials, functions, expressions)\n"
            "- Calculus (derivatives, integrals, limits, optimization)\n"
            "- Geometry (Euclidean geometry, trigonometry, coordinate geometry, spatial reasoning)\n"
            "- Statistics and probability (data analysis, distributions, hypothesis testing)\n"
            "- Number theory (prime numbers, divisibility, modular arithmetic)\n"
            "- Linear algebra (vectors, matrices, systems of equations, transformations)\n"
            "- Differential equations\n"
            "- Discrete mathematics (graph theory, combinatorics, logic)\n"
            "- Mathematical logic and proof techniques\n"
            "- Applied mathematics and mathematical modeling\n"
            "- Mathematical problem-solving and calculations\n\n"
            "IMPORTANT: Use the provided context from the knowledge base to answer questions accurately. "
            "Prioritize context when available, otherwise use your general knowledge.\n\n"
            "When refusing, respond with EXACTLY:\n"
            "\"I appreciate your question, but it doesn't relate to the subjects I specialize in. Please ask me about Physics, Mathematics, Chemistry, or Astronomy instead.\"\n\n"
            "For mathematics questions, provide:\n"
            "- Step-by-step solutions\n"
            "- Clear mathematical notation and formulas\n"
            "- Explanations of mathematical concepts and theorems\n"
            "- Proofs when applicable\n\n"
            "Be helpful and interpret questions broadly as mathematics-related when reasonable."
        ),
        "generation_config": {"temperature": 0.3, "top_p": 0.85, "top_k": 40, "max_output_tokens": 8192},
    },
    "chemistry": {
        "system": (
            "You are an expert Chemistry assistant specializing in all areas of chemistry.\n\n"
            "SCOPE: Answer questions about Chemistry topics including:\n"
            "- Chemical reactions and equations (balancing, stoichiometry, reaction types)\n"
            "- Organic chemistry (carbon compounds, reactions, mechanisms, functional groups)\n"
            "- Inorganic chemistry (elements, compounds, bonding, coordination chemistry)\n"
            "- Physical chemistry (thermodynamics, kinetics, electrochemistry, quantum chemistry)\n"
            "- Biochemistry (biological molecules, metabolic pathways, enzymes, proteins)\n"
            "- Analytical chemistry (methods, techniques, instrumentation)\n"
            "- Atomic structure and periodic table (trends, properties, electron configuration)\n"
            "- Chemical bonding and molecular structure (covalent, ionic, intermolecular forces)\n"
            "- Acids, bases, and pH (acid-base chemistry, buffers, titrations)\n"
            "- Applied chemistry and chemical processes\n\n"
            "IMPORTANT: Use the provided context from the knowledge base to answer questions accurately. "
            "Prioritize context when available, otherwise use your general knowledge.\n\n"
            "When refusing, respond with EXACTLY:\n"
            "\"I appreciate your question, but it doesn't relate to the subjects I specialize in. Please ask me about Physics, Mathematics, Chemistry, or Astronomy instead.\"\n\n"
            "For chemistry questions, provide:\n"
            "- Chemical equations and formulas\n"
            "- Clear explanations of chemical principles and mechanisms\n"
            "- Molecular structures when relevant\n"
            "- Proper chemical terminology and notation\n\n"
            "Be helpful and interpret questions broadly as chemistry-related when reasonable."
        ),
        "generation_config": {"temperature": 0.3, "top_p": 0.85, "top_k": 40, "max_output_tokens": 8192},
    },
    "astronomy": {
        "system": (
            "You are an expert Astronomy assistant specializing in all areas of astronomy and space science.\n\n"
            "SCOPE: Answer questions about Astronomy topics including:\n"
            "- Planets and planetary science (formation, composition, atmospheres, geology)\n"
            "- Stars and stellar evolution (life cycles, types, nuclear fusion, supernovae)\n"
            "- Galaxies and cosmology (structure, formation, expansion, dark matter/energy)\n"
            "- The solar system (planets, moons, asteroids, comets, formation)\n"
            "- Black holes and neutron stars (formation, properties, effects)\n"
            "- Space exploration and missions (spacecraft, missions, discoveries)\n"
            "- Celestial mechanics (orbits, gravity, motion)\n"
            "- Observational astronomy (telescopes, observation techniques, data)\n"
            "- Cosmology and the universe (Big Bang, structure, evolution, theories)\n"
            "- Astronomical phenomena (eclipses, meteor showers, transits, auroras)\n"
            "- Astrophysical processes and space physics\n\n"
            "IMPORTANT: Use the provided context from the knowledge base to answer questions accurately. "
            "Prioritize context when available, otherwise use your general knowledge.\n\n"
            "When refusing, respond with EXACTLY:\n"
            "\"I appreciate your question, but it doesn't relate to the subjects I specialize in. Please ask me about Physics, Mathematics, Chemistry, or Astronomy instead.\"\n\n"
            "For astronomy questions, provide:\n"
            "- Clear explanations of astronomical concepts\n"
            "- Current scientific understanding\n"
            "- Relevant data and measurements\n"
            "- Proper astronomical terminology\n\n"
            "Be helpful and interpret questions broadly as astronomy-related when reasonable."
        ),
        "generation_config": {"temperature": 0.3, "top_p": 0.85, "top_k": 40, "max_output_tokens": 8192},
    },
    "category_classifier": {
        "system": (
            "You are a category classification assistant. Your ONLY job is to identify which category "
            "a user's question belongs to.\n\n"
            "VALID CATEGORIES:\n"
            "- Physics: Questions about mechanics, thermodynamics, electromagnetism, quantum physics, optics, etc.\n"
            "- Mathematics: Questions about algebra, calculus, geometry, statistics, number theory, etc.\n"
            "- Chemistry: Questions about chemical reactions, organic/inorganic chemistry, biochemistry, etc.\n"
            "- Astronomy: Questions about planets, stars, galaxies, cosmology, space exploration, etc.\n\n"
            "INSTRUCTIONS:\n"
            "1. Analyze the user's question carefully.\n"
            "2. Determine if it matches ONE of the four categories above (Physics, Mathematics, Chemistry, Astronomy).\n"
            "3. If it matches a category, respond with EXACTLY this format:\n"
            "   \"This question belongs to: [Category Name]\"\n"
            "   (Replace [Category Name] with Physics, Mathematics, Chemistry, or Astronomy)\n\n"
            "4. If the question does NOT match any of the four categories, respond with EXACTLY:\n"
            "   \"This question is not relevant to any of the available categories. Please ask a question related to Physics, Mathematics, Chemistry, or Astronomy.\"\n\n"
            "CRITICAL RULES:\n"
            "- DO NOT answer the question itself, only classify it\n"
            "- DO NOT provide explanations beyond the category name\n"
            "- Use the EXACT response formats specified above\n"
            "- Be strict: if the question is about history, cooking, programming, etc., it is NOT relevant\n"
        ),
        "generation_config": {"temperature": 0.1, "top_p": 0.8, "top_k": 20, "max_output_tokens": 1024},
    },
    "default": {
        "system": (
            "You are an expert educational assistant specializing in Physics, Mathematics, Chemistry, and Astronomy.\n\n"
            "IMPORTANT: Use the provided context from the knowledge base to answer questions accurately. "
            "Prioritize context when available, otherwise use your general knowledge.\n\n"
            "Be helpful and generous in interpreting questions as relevant to your expertise."
        ),
        "generation_config": {"temperature": 0.3, "top_p": 0.85, "top_k": 40, "max_output_tokens": 8192},
    },
}

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Initialize knowledge base on startup
print("\n" + "="*50)
print("Initializing RAG-based Educational Assistant...")
print("="*50)
initialize_knowledge_base()
print("="*50 + "\n")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    # History format: [{role: "user"/"model", text: "..."}]
    history = data.get("history", [])
    mode = (data.get("mode") or "all_subjects").lower()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Pick profile (fallback to all_subjects)
    profile = PROFILES.get(mode, PROFILES["all_subjects"])
    system_instruction = profile["system"]
    generation_config = profile["generation_config"]

    # RAG: Retrieve relevant context from vector database
    retrieved_context = retrieve_relevant_context(
        user_message, vector_collection, mode=mode)

    # Build enhanced system instruction with context
    enhanced_system_instruction = system_instruction
    if retrieved_context:
        context_text = "\n\n".join(
            [f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(retrieved_context)])
        enhanced_system_instruction = (
            f"{system_instruction}\n\n"
            "RELEVANT CONTEXT FROM KNOWLEDGE BASE:\n"
            f"{context_text}\n\n"
            "Use the above context to provide accurate answers. If the context is relevant to the question, "
            "prioritize it in your response. If not, you can supplement with your general knowledge."
        )

    # Build chat history for Gemini (keep last 10 turns for context)
    contents = []
    for turn in history[-10:]:
        role = "user" if turn.get("role") == "user" else "model"
        contents.append({"role": role, "parts": [turn.get("text", "")]})
    # Prepend system/context as part of the latest user message for SDKs without system_instruction support
    combined_prompt = f"{enhanced_system_instruction}\n\nUser question:\n{user_message}"
    contents.append({"role": "user", "parts": [combined_prompt]})

    try:
        model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config=generation_config,
        )
        resp = model.generate_content(contents)

        # Get the complete response text
        if resp.text:
            text = resp.text
        elif resp.candidates and resp.candidates[0].content:
            # Fallback: get text from candidate if direct text access fails
            text = resp.candidates[0].content.parts[0].text if resp.candidates[
                0].content.parts else "(No response text)"
        else:
            text = "(No response text)"

        # Include retrieval info in response (for debugging)
        retrieval_info = {
            "retrieved_chunks": len(retrieved_context),
            "has_context": len(retrieved_context) > 0
        }

        return jsonify({
            "reply": text,
            "mode": mode,
            "rag_info": retrieval_info
        })
    except Exception as e:
        # Log error details and return a friendly reply to the client
        error_msg = str(e)
        print("Error generating response:\n" + traceback.format_exc())
        # Send HTTP 200 with a reply so the frontend can display it gracefully
        return jsonify({
            "reply": f"Sorry, I couldn’t complete that request. Details: {error_msg}",
            "mode": mode,
            "rag_info": {
                "retrieved_chunks": len(retrieved_context) if 'retrieved_context' in locals() else 0,
                "has_context": bool(retrieved_context) if 'retrieved_context' in locals() else False
            }
        })


@app.post("/reload-documents")
def reload_documents():
    """Reload documents from the documents folder into the vector database."""
    global vector_collection
    try:
        # Clear existing collection
        chroma_client.delete_collection(name=collection_name)
        vector_collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Reload documents
        documents_folder = os.path.join(os.path.dirname(__file__), "documents")
        documents = load_documents_from_folder(documents_folder)

        if documents:
            process_and_store_documents(documents, vector_collection)
            return jsonify({
                "status": "success",
                "message": f"Reloaded {len(documents)} documents",
                "chunks": vector_collection.count()
            })
        else:
            return jsonify({
                "status": "warning",
                "message": "No documents found in documents folder"
            })
    except (IOError, ValueError, RuntimeError) as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.get("/health")
def health():
    db_status = {
        "vector_db": "connected",
        "collection_count": vector_collection.count(),
        "embedding_model": EMBEDDING_MODEL
    }
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rag_status": db_status
    })


if __name__ == "__main__":
    # WARNING: debug=True is for local dev only
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
