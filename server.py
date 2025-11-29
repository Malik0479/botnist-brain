import os
import shutil
from flask import Flask, request, jsonify
from pyngrok import ngrok, conf
from supabase import create_client, Client

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# Import Settings to optionally clean up logs (optional but recommended)
from chromadb.config import Settings

# ---------------- CONFIGURATION ---------------- #

# NGROK Setup
NGROK_AUTH_TOKEN = "2uLhzLCZ0wy53qDDxwmECyYe8wa_4DgFujSh9gqL2Yw5oNYw7"
conf.get_default().auth_token = NGROK_AUTH_TOKEN

# SUPABASE Setup
SUPABASE_URL = "https://hhavudpqysmahejbeqrb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhoYXZ1ZHBxeXNtYWhlamJlcXJiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NDExMzM0MSwiZXhwIjoyMDc5Njg5MzQxfQ.4T1OEWTkOUZ4Tke5x-nbpWEFgndlIbXa5MGela64FcU"
BUCKET_NAME = "scrapes"

# GOOGLE AI Setup
GOOGLE_API_KEY = "AIzaSyBZwMFSKnSRG2LZHEEu2g020R0126T_DEM"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY 

# Initialize Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Using the requested model versions
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.3,
    convert_system_message_to_human=True
)

app = Flask(__name__)

CHROMA_PATH = "./chroma_db"
os.makedirs(CHROMA_PATH, exist_ok=True)

# ---------------- HELPER FUNCTIONS ---------------- #

def get_supabase_history(user_id, session_id, token_hash):
    """
    Fetches history matching user_id, session_id AND token_hash.
    """
    try:
        response = supabase.table("chat_history") \
            .select("messages") \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .eq("token_hash", token_hash) \
            .execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]['messages']
        return []
    except Exception as e:
        print(f"⚠️ Error fetching history: {e}")
        return []

def update_supabase_history(user_id, session_id, token_hash, user_msg, bot_msg):
    """
    Updates history. Enforces a limit of 10 messages (sliding window).
    """
    try:
        current_history = get_supabase_history(user_id, session_id, token_hash)
        
        # Append new interaction
        current_history.append({"role": "user", "text": user_msg})
        current_history.append({"role": "bot", "text": bot_msg})

        # --- Limit History Size ---
        # If history exceeds 10 messages, keep only the last 10 (most recent)
        if len(current_history) > 10:
            current_history = current_history[-10:]

        data = {
            "user_id": user_id, 
            "session_id": session_id,
            "token_hash": token_hash, # ✅ Included in update
            "messages": current_history
        }
        
        # Upsert based on the unique constraint (usually user_id + session_id)
        supabase.table("chat_history").upsert(data, on_conflict="user_id, session_id").execute()
        
    except Exception as e:
        print(f"⚠️ Error updating history: {e}")

def get_chroma_db(key):
    directory = f"{CHROMA_PATH}/{key}"
    if not os.path.exists(directory):
        return None
    return Chroma(
        persist_directory=directory, 
        embedding_function=embeddings,
        client_settings=Settings(anonymized_telemetry=False)
    )

# ---------------- API ENDPOINTS ---------------- #

@app.route("/ingest", methods=["POST"])
def ingest_data():
    payload = request.get_json()
    
    # 1. Handle Supabase Webhook Format
    # Supabase sends: { "type": "INSERT", "record": { "job_hash": "...", ... } }
    if "record" in payload:
        unique_hash = payload["record"].get("job_hash")
    else:
        # Handle direct manual calls
        unique_hash = payload.get("unique_hash")

    if not unique_hash:
        print("❌ No hash found in payload")
        return jsonify({"error": "Missing unique_hash"}), 400

    print(f"⚡ Auto-Ingesting for Hash: {unique_hash}")

    # 2. Define Paths
    file_name = f"{unique_hash}.txt" # We are using .txt now
    local_path = f"./{file_name}"

    try:
        # 3. Download from Supabase Storage
        print(f"   Downloading {file_name}...")
        with open(local_path, 'wb+') as f:
            res = supabase.storage.from_(BUCKET_NAME).download(file_name)
            f.write(res)
        
        # 4. Load & Split
        loader = TextLoader(local_path, encoding="utf8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_splits = text_splitter.split_documents(documents)

        # 5. Create Isolated Vector Store (The "Key" Logic)
        # We create a folder named exactly after the hash.
        persist_dir = f"{CHROMA_PATH}/{unique_hash}"
        
        # Clear old data if it exists (re-training)
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        Chroma.from_documents(
            documents=all_splits, 
            embedding=embeddings, 
            persist_directory=persist_dir,
            client_settings=Settings(anonymized_telemetry=False)
        )
        
        # Cleanup
        os.remove(local_path)
        print(f"✅ Ingestion Complete for {unique_hash}")
        return jsonify({"message": "Ingestion successful", "hash": unique_hash})

    except Exception as e:
        if os.path.exists(local_path): os.remove(local_path)
        print(f"❌ Ingest Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def process_query():
    data = request.get_json()
    query_text = data.get("query")
    unique_hash = data.get("unique_hash")
    user_id = data.get("user_id")
    token_hash = data.get("token_hash") # ✅ Get token_hash from request

    if not query_text or not unique_hash or not user_id or not token_hash:
        return jsonify({"error": "Missing query, unique_hash, user_id, or token_hash"}), 400

    if query_text.lower() == "resethistory":
        try:
            supabase.table("chat_history")\
                .delete()\
                .eq("user_id", user_id)\
                .eq("session_id", unique_hash)\
                .execute()
            return jsonify({"response": "History reset."})
        except Exception as e:
             return jsonify({"error": str(e)}), 500

    # 1. Get Vector Store
    vectordb = get_chroma_db(unique_hash)
    if not vectordb:
        return jsonify({"response": "No data found for this hash. Please ingest first."}), 404

    # 2. Get History (passing token_hash)
    history_messages = get_supabase_history(user_id, unique_hash, token_hash)
    
    # Format for Prompt
    formatted_history = "\n".join([f"{msg['role']}: {msg['text']}" for msg in history_messages])

    # 3. Create Modern Chain
    system_prompt = (
        "You are a dedicated and professional Customer Support Agent for the store represented in the provided context. "
        "Your goal is to assist customers accurately, politely, and efficiently based *only* on the information available to you."
        "\n\n"
        "Guidelines:"
        "\n1. **Context First**: Base your answers strictly on the 'Context' provided below. Do not make up products, policies, or details not found in the text."
        "\n2. **Tone**: Maintain a friendly, helpful, and empathetic tone. Use professional language."
        "\n3. **History Awareness**: Use the 'Conversation History' to understand follow-up questions and maintain continuity."
        "\n4. **Fallback**: If the answer to the asked question is not in the context, politely admit you don't have that information and suggest they contact human support. Do this if you don't have the answer for the ASKED QUESTION in the context. Do not hallucinate answers."
        "\n5. **Conciseness**: Keep answers direct and relevant to the user's query."
        "\n\n"
        "Conversation History:\n"
        f"{formatted_history}"
        "\n\n"
        "Context from Store Data:\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vectordb.as_retriever(), question_answer_chain)

    try:
        # Run chain
        result = rag_chain.invoke({"input": query_text})
        response = result["answer"]

        # Save History (includes token_hash and limit logic)
        update_supabase_history(user_id, unique_hash, token_hash, query_text, response)

        return jsonify({"response": response})

    except Exception as e:
        print(f"❌ Query Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    public_url = ngrok.connect(5000)
    print(f"🔗 Ngrok tunnel established: {public_url}")
    app.run(host="0.0.0.0", port=5000)