from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import os,sys
import json
import sqlite3


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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    # Ensure database is initialized before anything else
    init_db()
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
        persist_directory=persist_directory,
    )
    # print("Vector store initialized.")

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
        # print("Documents added to vector store and persisted.")

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
    - Total Belanja: ([item1(quantity x price), ...]\n) + RpXXX (city_name) = RpXXX

    If the input contains only a city_name, respond with "Shipping Fee: RpXXX (destination: city_name)".

    If the city_name is not listed, respond with "Area pengiriman diluar JABODETABEK kami akan kenakan cas Rp.10.000 biaya tambahan pengiriman".

    If the number currency Rp use comma for digit number 
    """

    # Initialize the prompt template
    rag_prompt = PromptTemplate.from_template(template)
    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough() }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    question = sys.argv[1]
    retrieved_docs = retriever.invoke({"input": question})
    formatted_context = format_docs(retrieved_docs) if retrieved_docs else "No relevant information found."
    answer = rag_chain.invoke(question)
    print(answer)
    # print({"question": question, "answer": answer})





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

if __name__ == "__main__":
    # Login ke Hugging Face secara interaktif
    main()