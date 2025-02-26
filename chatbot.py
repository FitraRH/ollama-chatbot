from flask import Flask, request, jsonify, render_template_string
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import os
import json
import sqlite3

app = Flask(__name__)

# File paths for persistent storage
db_path = "store.db"
model_config_path = "./model_config.json"

# Ensure the persistence directory exists
persist_directory = "./chroma_langchain_db"
os.makedirs(persist_directory, exist_ok=True)

def init_db():
    """Initialize SQLite database with initial data"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create barang (products) table
    c.execute('''CREATE TABLE IF NOT EXISTS barang (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nama TEXT,
                    harga INTEGER,
                    kategori TEXT,
                    ukuran TEXT,
                    stok INTEGER)''')
    
    # Create ongkir (shipping) table
    c.execute('''CREATE TABLE IF NOT EXISTS ongkir (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kota TEXT,
                    biaya INTEGER)''')
    
    # Check if barang table is empty and insert initial data
    c.execute("SELECT COUNT(*) FROM barang")
    if c.fetchone()[0] == 0:
        barang = [
            ("Baju Kemeja", 100000, "Pakaian", "S,M,L,XL", 1),
            ("Celana Cino", 180000, "Pakaian", "M,L,XL", 1),
            ("Topi Kinz", 50000, "Aksesoris", "All Size", 0)
        ]
        c.executemany("INSERT INTO barang (nama, harga, kategori, ukuran, stok) VALUES (?, ?, ?, ?, ?)", barang)
    
    # Check if ongkir table is empty and insert initial data
    c.execute("SELECT COUNT(*) FROM ongkir")
    if c.fetchone()[0] == 0:
        ongkir = [
            ("Jakarta", 20000),
            ("Bandung", 15000),
            ("Surabaya", 25000),
            ("Luar Kota", 45000)
        ]
        c.executemany("INSERT INTO ongkir (kota, biaya) VALUES (?, ?)", ongkir)
    
    conn.commit()
    conn.close()

# Ensure database is initialized before anything else
init_db()

def get_barang():
    """Retrieve products from the database"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT nama, harga, kategori, ukuran, stok FROM barang")
    data = [{"nama": row[0], "harga": row[1], "kategori": row[2], "ukuran": row[3].split(','), "stok": bool(row[4])} for row in c.fetchall()]
    conn.close()
    return data

def get_ongkir():
    """Retrieve shipping rates from the database"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT kota, biaya FROM ongkir")
    data = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    return data

# Initialize or load the LLM
if os.path.exists(model_config_path):
    with open(model_config_path, "r") as f:
        config = json.load(f)
    llm = ChatOllama(**config)
    print("Model initialized from configuration file.")
else:
    llm = ChatOllama(
        model="modellexnew:latest",
        temperature=0,
    )
    config = {"model": "modellexnew:latest", "temperature": 0}
    with open(model_config_path, "w") as f:
        json.dump(config, f)
    print("Model initialized and configuration saved.")

# Initialize the embeddings
embeddings = FastEmbedEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

# Initialize the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=persist_directory,
)
print("Vector store initialized.")

# Retrieve current barang and ongkir data
barang = get_barang()
ongkir = get_ongkir()

# Membuat string daftar barang untuk dokumen
barang_text = "\n".join([
    f"{item['nama']} (Kategori: {item['kategori']}, Harga: Rp{item['harga']}, "
    f"Ukuran: {', '.join(item['ukuran'])}, Stok: {'Tersedia' if item['stok'] else 'Habis'})"
    for item in barang
])
ongkir_text = ", ".join([f"{k} (Rp{v})" for k, v in ongkir.items()])

# Membuat dokumen
documents = [
    Document(
        page_content=f"Barang yang tersedia:\n{barang_text}.",
        metadata={"source": "product_info"},
    ),
    Document(
        page_content=f"Ongkos kirim: {ongkir_text}.",
        metadata={"source": "shipping_info"},
    ),
    Document(
        page_content="Setelah memilih warna, ukuran, dan alamat, silakan lakukan transfer sesuai total biaya.",
        metadata={"source": "cart"},
    ),
]

# Tambahkan dokumen ke vector store jika belum ada data
if not os.listdir(persist_directory):
    vector_store.add_documents(documents)
    print("Documents added to vector store and persisted.")

# Set up retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Define prompt template
template = """
You are an assistant for calculating the total cost of items in a shopping cart including shipping costs.
Here is the list of available items, their prices, categories, sizes, stock status, and shipping fees:
{context}

The user has provided the following information:
{question}

If the input contains item details, calculate the total price and return it in this format:
- Details: [item1 (quantity x price), item2 (quantity x price), ...]
- Shipping Fee: RpXXX (destination: city_name)
- Stock Info: [item1: Available, item2: Out of Stock, ...]
- Total Belanja: [item1(quantity x price), ...] + RpXXX (city_name) = RpXXX

If the input contains only a city_name, respond with "Shipping Fee: RpXXX (destination: city_name)".

If the city_name is not listed, respond with "Area pengiriman diluar JABODETABEK kami akan kenakan cas Rp.10.000 biaya tambahan pengiriman".
"""

# Initialize the prompt template
rag_prompt = PromptTemplate.from_template(template)

# Function to format the documents into a string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# HTML template (unchanged from previous version)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Shopping Assistant</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            background-color: #f0f2f5;
        }
        .container { 
            background-color: white; 
            padding: 20px; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .product-list, .shipping-rates { 
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 6px;
        }
        h1 { 
            color: #1a73e8;
            text-align: center;
        }
        h2 {
            color: #202124;
            margin-top: 20px;
        }
        form { 
            margin: 20px 0;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        input[type="text"] { 
            width: 100%; 
            padding: 12px;
            border: 1px solid #dadce0;
            border-radius: 4px;
            font-size: 16px;
            box-sizing: border-box;
        }
        input[type="submit"] { 
            padding: 12px 24px;
            background-color: #1a73e8;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            align-self: flex-start;
        }
        input[type="submit"]:hover { 
            background-color: #1557b0;
        }
        .response { 
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f0fe;
            border-left: 4px solid #1a73e8;
            border-radius: 4px;
        }
        ul {
            list-style-type: none;
            padding-left: 0;
        }
        li {
            margin: 10px 0;
            padding: 10px;
            background-color: white;
            border-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .example {
            color: #5f6368;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Shopping Assistant</h1>
        
        <div class="product-list">
            <h2>Available Products:</h2>
            <ul>
            {% for item in available_items %}
                <li>
                    <strong>{{ item.nama }}</strong> ({{ item.kategori }})
                    <ul>
                        <li>Price: Rp{{ item.harga }}</li>
                        <li>Sizes: {{ ", ".join(item.ukuran) }}</li>
                        <li>Stock: {{ "Available" if item.stok else "Out of Stock" }}</li>
                    </ul>
                </li>
            {% endfor %}
            </ul>
        </div>

        <div class="shipping-rates">
            <h2>Shipping Rates:</h2>
            <ul>
            {% for city, rate in shipping_rates.items() %}
                <li><strong>{{ city }}</strong>: Rp{{ rate }}</li>
            {% endfor %}
            </ul>
        </div>

        <form method="POST">
            <h2>Ask a Question:</h2>
            <p class="example">Example: "What is the shipping cost to Jakarta for 2 Baju Kemeja size M?"</p>
            <input type="text" name="question" placeholder="Enter your question here" required>
            <input type="submit" value="Ask">
        </form>

        {% if answer %}
        <div class="response">
            <h3>Response:</h3>
            <p>{{ answer | replace('\n', '<br>') | safe }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    available_items = get_barang()
    shipping_rates = get_ongkir()

    if request.method == "POST":
        question = request.form.get("question")
        if question:
            # Retrieve documents
            retrieved_docs = retriever.invoke({"input": question})
            formatted_context = format_docs(retrieved_docs) if retrieved_docs else "No relevant information found."
            
            # Get the answer from the chain
            answer = rag_chain.invoke(question)
    
    return render_template_string(
        HTML_TEMPLATE,
        available_items=available_items,
        shipping_rates=shipping_rates,
        answer=answer
    )

# Keep the favicon route
@app.route('/favicon.ico')
def favicon():
    return '', 204

# Keep the /ask endpoint for API access
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question")
        if not question:
            return jsonify({"error": "Question field is required."}), 400

        retrieved_docs = retriever.invoke({"input": question})
        formatted_context = format_docs(retrieved_docs) if retrieved_docs else "No relevant information found."
        answer = rag_chain.invoke(question)
        return jsonify({"question": question, "answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
