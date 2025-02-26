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
model_config_path = "./model.json"

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
    c.execute("SELECT id, nama, harga, kategori, ukuran, stok FROM barang")
    data = [{"id": row[0], "nama": row[1], "harga": row[2], "kategori": row[3], "ukuran": row[4].split(','), "stok": bool(row[5])} for row in c.fetchall()]
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
        model="hf.co/ojisetyawan/gemma2-9b-cpt-sahabatai-v1-instruct-Q4_K_M-GGUF:latest",
        temperature=0,
    )
    config = {"model": "hf.co/ojisetyawan/gemma2-9b-cpt-sahabatai-v1-instruct-Q4_K_M-GGUF:latest", "temperature": 0}
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
    # persist_directory=persist_directory,
)
print("Vector store initialized.")

# Retrieve current barang and ongkir data
barang = get_barang()
ongkir = get_ongkir()

# Membuat string daftar barang untuk dokumen
barang_text = "\n".join([f"{item['nama']} (Kategori: {item['kategori']}, Harga: Rp{item['harga']}, "
    f"Ukuran: {', '.join(item['ukuran'])}, Stok: {'Tersedia' if item['stok'] else 'Habis'})"
    for item in barang])
ongkir_text = ", ".join([f"{k.lower()} (Rp{v})" for k, v in ongkir.items()])

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

# Define prompt template for customer service toko 
# template = """
#     You are an assistant for calculating the total cost of items in a shopping cart including shipping costs.
#     Here is the list of available items, their prices, categories, sizes, stock status, and shipping fees:
#     {context}

#     The user has provided the following information:
#     {question}

#     If the input contains item details, calculate the total price and return it in this format:
#     - Details: [item1 (quantity x price) (sizes), item2 (quantity x price) (sizes), ...]
#     - Shipping Fee: RpXXX (destination: city_name)
#     - Stock Info: [item1: Available, item2: Out of Stock, ...]
#     - Total Belanja: ([item1(quantity x price), ...]\n) + RpXXX (city_name) = RpXXX

#     If the input contains only a city_name, respond with "Shipping Fee: RpXXX (destination: city_name)".

#     If the number currency Rp use comma for digit number 

#     """

template = """You are an assistant to calculate the total cost of items in the shopping cart including shipping costs.
Here is a list of available items, prices, categories, sizes, stock status, and shipping costs:
{context}

The user has provided the following information:
{question}

If the input contains item details, calculate the total price and return it in this format:
- Details: [item1 (quantity x price) (size), item2 (quantity x price) (size), ...]
- Shipping Cost: RpXXX (destination: city_name)
- Stock Info: [item1: In Stock, item2: Out of Stock, ...]
- Total Shopping: ([item1 (quantity x price), ...]\n) + RpXXX (city_name) = RpXXX

If the input only contains city_name, answer with "Shipping Cost: RpXXX (destination: city_name)".

If the currency number is Rp, use a comma for the digits

If you ask for the size and the size is available with the size per item, please reply with "Item size available" if not "Item size not available"

If only item: Available, "item stock available" if not "item not available"""

# Initialize the prompt template
rag_prompt = PromptTemplate.from_template(template)

# Function to format the documents into a string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough() }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Function to calculate discount
def apply_discount(total, item_count):
    if item_count > 3:
        return total * 0.9  # 10% discount for more than 3 items
    return total

# Function for product search
def search_product(query):
    products = get_barang()
    result = [p for p in products if query.lower() in p["nama"].lower()]
    return result

# Function to notify admin for low stock
def check_low_stock():
    low_stock_items = []
    for item in get_barang():
        if item['stok'] < 2:  # threshold for low stock
            low_stock_items.append(item)
    return low_stock_items

# HTML template (updated with a search bar)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Shopping Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
</head>
<body>
    <div class="container my-5">
        <h1 class="text-center mb-4">Shopping Assistant</h1>
        
        <div class="product-list">
            <h2 class="mb-3">Available Products</h2>
            <table class="table table-striped table-bordered">
                <thead class="table-dark">
                    <tr>
                        <th>Name</th>
                        <th>Category</th>
                        <th>Price</th>
                        <th>Sizes</th>
                        <th>Stock</th>
                    </tr>
                </thead>
                <tbody>
                {% for item in available_items %}
                    <tr>
                        <td>{{ item.nama }}</td>
                        <td>{{ item.kategori }}</td>
                        <td>Rp{{ item.harga }}</td>
                        <td>{{ ", ".join(item.ukuran) }}</td>
                        <td>{{ "Available" if item.stok else "Out of Stock" }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="shipping-rates mt-5">
            <h2 class="mb-3">Shipping Rates</h2>
            <ul class="list-group">
            {% for city, rate in shipping_rates.items() %}
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    <strong>{{ city }}</strong>
                    <span class="badge bg-primary">Rp{{ rate }}</span>
                </li>
            {% endfor %}
            </ul>
        </div>

        <div class="mt-5">
            <h2>Search Products</h2>
            <form method="POST" class="row g-3">
                <div class="col-md-8">
                    <input type="text" name="search_query" class="form-control" placeholder="Search by name" required>
                </div>
                <div class="col-md-4">
                    <button type="submit" class="btn btn-primary w-100">Search</button>
                </div>
            </form>
        </div>

        <div class="mt-4">
            <h2>Ask a Question</h2>
            <p class="text-muted">Example: "What is the shipping cost to Jakarta for 2 Baju Kemeja size M?"</p>
            <form method="POST" class="row g-3">
                <div class="col-md-8">
                    <input type="text" name="question" class="form-control" placeholder="Enter your question here" required>
                </div>
                <div class="col-md-4">
                    <button type="submit" class="btn btn-success w-100">Ask</button>
                </div>
            </form>
        </div>

        {% if search_results %}
        <div class="response mt-5">
            <h3>Search Results</h3>
            <ul class="list-group">
                {% for result in search_results %}
                    <li class="list-group-item"> <strong>{{ result.nama }}</strong> - Rp{{ result.harga }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        {% if answer %}
        <div class="response mt-4">
            <h3>Response</h3>
            <p class="alert alert-info">{{ answer | replace('\n', '<br>') | safe }}</p>
        </div>
        {% endif %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz" crossorigin="anonymous"></script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    available_items = get_barang()
    shipping_rates = get_ongkir()
    search_results = []

    if request.method == "POST":
        search_query = request.form.get("search_query")
        if search_query:
            search_results = search_product(search_query)
        
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
        search_results=search_results,
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
    app.run(debug=True)