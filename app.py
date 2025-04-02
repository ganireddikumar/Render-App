import os
import uuid
from flask import Flask, request, jsonify, session
from datetime import datetime
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
import weaviate
from weaviate.classes.query import Filter
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure, Tokenization
from langchain_together import TogetherEmbeddings, Together
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Weaviate
from pypdf import PdfReader
from docx import Document

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'gani')

# Configure CORS properly
CORS(app)

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Weaviate client
def get_weaviate_client():
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv('WEAVIATE_URL'),
            auth_credentials=Auth.api_key(os.getenv('WEAVIATE_API_KEY'))
        )
        # Set timeout for operations
        client.timeout_config = (60, 300)  # (connect_timeout, read_timeout) in seconds
        return client
    except Exception as e:
        print(f"Error connecting to Weaviate: {e}")
        raise

# Replace the direct client initialization with the function
weaviate_client = get_weaviate_client()

# Initialize embeddings and llm
embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-32k-retrieval",
    together_api_key=os.getenv('TOGETHER_API_KEY')
)

llm = Together(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    temperature=0.1,
    together_api_key=os.getenv('TOGETHER_API_KEY')
)

# Helper functions
def get_weaviate_client():
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv('WEAVIATE_URL'),
        auth_credentials=Auth.api_key(os.getenv('WEAVIATE_API_KEY'))
    )

def get_mysql_connection():
    try:
        ca_cert = os.getenv('TIDB_CA_CERT')
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pem') as temp_cert:
            temp_cert.write(ca_cert)
            temp_cert_path = temp_cert.name

        conn = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST'),
            port=int(os.getenv('MYSQL_PORT')),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD'),
            database=os.getenv('MYSQL_DB'),
            ssl_ca=temp_cert_path
        )
        
        os.unlink(temp_cert_path)
        return conn
    except Error as e:
        print(f"MySQL Error: {e}")
        return None

def process_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                try:
                    text += page.extract_text() + "\n"
                except Exception as e:
                    print(f"Error extracting page: {e}")
                    continue
            return text.strip()
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise Exception("Failed to process PDF file")

def process_docx(file_path):
    try:
        doc = Document(file_path)
        text = []
        for para in doc.paragraphs:
            if para.text.strip():
                text.append(para.text)
        return "\n".join(text)
    except Exception as e:
        print(f"Error processing DOCX: {e}")
        raise Exception("Failed to process DOCX file")

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def initialize_weaviate_schema():
    with get_weaviate_client() as weaviate_client:
        if not weaviate_client.collections.exists("DocumentChunks"):
            weaviate_client.collections.create(
                name="DocumentChunks",
                properties=[
                    {
                        "name": "document_id",
                        "data_type": DataType.TEXT,
                        "description": "The document identifier"
                    },
                    {
                        "name": "chunk",
                        "data_type": DataType.TEXT,
                        "description": "The text chunk content"
                    },
                    {
                        "name": "chunk_index",
                        "data_type": DataType.INT,
                        "description": "The index of the chunk in the document"
                    }
                ],
                vectorizer_config=None,
                description="Collection for storing document chunks"
            )

def initialize_database():
    conn = get_mysql_connection()
    if not conn:
        print("Failed to connect to database")
        return False

    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(10) NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INT AUTO_INCREMENT PRIMARY KEY,
                document_id INT NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_index INT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                document_id INT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        
        conn.commit()
        print("Database initialized successfully")
        return True
    except Error as e:
        print(f"Error initializing database: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

# API Endpoints
@app.route('/api/upload', methods=['POST'])
def upload_file():
    # ... existing validation code ...

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            if filename.lower().endswith('.pdf'):
                text = process_pdf(file_path)
            else:
                text = process_docx(file_path)
        except Exception as e:
            return jsonify({"error": f"File processing error: {str(e)}"}), 500

        chunks = chunk_text(text)
        
        # Process in smaller batches
        BATCH_SIZE = 10
        total_chunks = len(chunks)
        
        conn = get_mysql_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
            
        cursor = conn.cursor()
        weaviate_client = None
        try:
            cursor.execute(
                "INSERT INTO documents (filename, file_type) VALUES (%s, %s)",
                (filename, os.path.splitext(filename)[1])
            )
            document_id = cursor.lastrowid

            # Process chunks in batches
            for i in range(0, total_chunks, BATCH_SIZE):
                batch_chunks = chunks[i:i + BATCH_SIZE]
                batch_vectors = embeddings.embed_documents(batch_chunks)
                
                weaviate_client = get_weaviate_client()
                with weaviate_client:
                    collection = weaviate_client.collections.get("DocumentChunks")
                    for j, (chunk, vector) in enumerate(zip(batch_chunks, batch_vectors)):
                        chunk_index = i + j
                        collection.data.insert(
                            properties={
                                "document_id": str(document_id),
                                "chunk": chunk,
                                "chunk_index": chunk_index
                            },
                            vector=vector
                        )

                    # Insert chunks into MySQL
                    for j, chunk in enumerate(batch_chunks):
                        chunk_index = i + j
                        cursor.execute(
                            "INSERT INTO document_chunks (document_id, chunk_text, chunk_index) VALUES (%s, %s, %s)",
                            (document_id, chunk, chunk_index)
                        )
                    
                    conn.commit()
                
                if weaviate_client:
                    weaviate_client.close()

            return jsonify({
                "message": "File processed successfully",
                "document_id": document_id,
                "chunk_count": total_chunks
            })

        except Exception as e:
            conn.rollback()
            return jsonify({"error": str(e)}), 500
        finally:
            cursor.close()
            conn.close()
            if weaviate_client:
                weaviate_client.close()

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'question' not in data or 'document_id' not in data:
        return jsonify({"error": "Missing question or document_id"}), 400
    
    if 'chat_history' not in session:
        session['chat_history'] = []

    try:
        question_embedding = embeddings.embed_query(data['question'])

        weaviate_client = get_weaviate_client()
        try:
            with weaviate_client:
                collection = weaviate_client.collections.get("DocumentChunks")
                response = collection.query.near_vector(
                    near_vector=question_embedding,
                    distance=0.85,
                    limit=5,
                    filters=Filter.by_property("document_id").equal(str(data['document_id'])),
                    return_properties=["chunk", "chunk_index"]
                )
                
                sorted_chunks = sorted(response.objects, key=lambda x: x.properties["chunk_index"])
                context = "\n\n".join([obj.properties["chunk"] for obj in sorted_chunks])
        finally:
            weaviate_client.close()
            
        if not context.strip():
            return jsonify({"answer": "I don't have enough information to answer that question."})

        prompt = f"""You are a helpful AI assistant analyzing a document. Use the provided context to answer the question thoroughly and accurately. 
    
        Context: {context}
    
        Question: {data['question']}
        
        Instructions:
        1. Use the information from the provided context to answer the question
        2. If you can make a reasonable inference from the context, you may do so while indicating it's an inference
        3. If the context is partially relevant, provide what information you can and explain what's missing
        4. Only say "I don't have enough information" if the context is completely unrelated to the question
        5. Keep your answer clear and well-structured
        
        Answer: """
        
        answer = llm(prompt)

        conn = get_mysql_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO conversations (document_id, question, answer) VALUES (%s, %s, %s)",
                    (data['document_id'], data['question'], answer)
                )
                conn.commit()
            except Exception as db_error:
                print(f"Error storing conversation: {str(db_error)}")
            finally:
                cursor.close()
                conn.close()

        chat_entry = {
            'document_id': data['document_id'],
            'question': data['question'],
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }
        session['chat_history'].append(chat_entry)
        
        return jsonify({
            "answer": answer,
            "chat_history": session['chat_history']
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    conn = get_mysql_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM documents ORDER BY upload_date DESC")
        documents = cursor.fetchall()
        return jsonify(documents)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    document_id = request.args.get('document_id')
    if not document_id:
        return jsonify({"error": "Missing document_id parameter"}), 400

    conn = get_mysql_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM conversations WHERE document_id = %s ORDER BY timestamp DESC",
            (document_id,)
        )
        conversations = cursor.fetchall()
        return jsonify(conversations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the RAG Application API'})

@app.route('/api/clear-chat-history', methods=['POST'])
def clear_chat_history():
    if 'chat_history' in session:
        session.pop('chat_history')
    return jsonify({"message": "Chat history cleared successfully"})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    initialize_weaviate_schema()
    initialize_database()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
