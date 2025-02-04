# Shopping Assistant Flask Application

This Flask application provides a simple shopping assistant with the ability to answer questions about products and shipping fees. It uses an AI model for answering questions, integrates with a vector store for document retrieval, and supports both a web interface and an API endpoint.

## Features:
1. **Product Information**: Displays a list of available products, including their prices, sizes, stock status, and categories.
2. **Shipping Rates**: Shows available shipping cities and their respective fees.
3. **AI Question Answering**: Allows users to ask questions regarding the total cost of shopping, including items and shipping fees. It calculates and responds with the total cost based on available stock and shipping information.
4. **API Access**: Provides a `/ask` endpoint for API access to get answers to questions in JSON format.

## Requirements:
1. **Ollama Model**: The application relies on the Ollama model for question answering. You must first pull the model before running the application.
2. **Flask**: For serving the web application.
3. **SQLite**: For local database storage (for storing products and shipping rates).
4. **LangChain**: For document retrieval and question answering.
5. **Chroma**: For vector store storage and document indexing.

### Prerequisites:
1. **Windows Subsystem for Linux (WSL)**: If you're using Windows, it's recommended to run the application in **WSL** with **VSCode**.
2. **Ollama Installation**: Ollama needs to be installed for the AI model to work. This is a prerequisite for running the application.

---

## Setup Instructions:

### 1. Install Ollama (For Windows Users):
First, you'll need to install **Ollama**. To do so:
- Open **WSL** (Windows Subsystem for Linux) in your VSCode terminal.
- Install **Ollama** by running the following commands in the terminal:
    ```bash
    curl -sSL https://ollama.com/install.sh | sudo bash
    ```

### 2. Pull the Ollama Model:
Before starting the application, pull the required model using Ollama:
- Run the following command to pull the model:
    ```bash
    ollama pull hf.co/ojisetyawan/gemma2-9b-cpt-sahabatai-v1-instruct-Q4_K_M-GGUF
    ```

> **Note**: Make sure that you have sufficient disk space and an internet connection to download the model.

---

### 3. Install Dependencies:
Ensure that you have the required Python libraries installed by running the following commands:

```bash
pip install flask langchain langchain_community ollama chromadb sqlite3
```

**Optional (Recommended)**: It's highly recommended to use a **virtual environment** to isolate the dependencies for this project.

#### 3.1 Create and Activate a Virtual Environment:
If you're not already using a virtual environment, create one as follows:

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows (WSL)**:
    ```bash
    source venv/bin/activate
    ```

- **macOS/Linux**:
    ```bash
    source venv/bin/activate
    ```

Once the virtual environment is active, install the dependencies:

```bash
pip install flask langchain langchain_community ollama chromadb sqlite3
```

#### 3.2 Deactivate Virtual Environment:
When you’re done working, you can deactivate the virtual environment using:

```bash
deactivate
```

---

### 4. Set up the Database:
The application uses an SQLite database to store product and shipping information. When you first run the application, the `store.db` file will be created, and initial data will be populated into the `barang` (products) and `ongkir` (shipping) tables.

---

### 5. Run the Application:
To start the application, run:

```bash
python chatbot.py
```

By default, the application will start on `http://127.0.0.1:5000/` in your browser.

---

### 6. Using the Web Interface:
1. **Home Page**: The main page displays the available products and shipping rates. You can ask questions regarding the total cost of shopping, including product quantity and shipping.
2. **Ask a Question**: You can enter a question in the provided input field. For example: 
    - "What is the shipping cost to Jakarta for 2 Baju Kemeja size M?"
    - "How much is the total cost if I buy 1 Baju Kemeja and 2 Celana Cino?"

The application will respond with:
- The product details (quantity, price)
- The shipping fee
- The total price

---

## API Endpoint:

**POST** `/ask`
- **Request body**: JSON object containing the `question`.
    Example:
    ```json
    {
        "question": "What is the shipping cost to Surabaya for 1 Topi Kinz?"
    }
    ```

- **Response**: JSON object with the answer to the question.
    Example:
    ```json
    {
        "question": "What is the shipping cost to Surabaya for 1 Topi Kinz?",
        "answer": "Shipping Fee: Rp25000 (destination: Surabaya)"
    }
    ```

---

## Structure of the Code:

1. **Flask Web Application**: The core web application is built using Flask. It serves the main page and the `/ask` endpoint.
2. **Database**: Uses SQLite to store product data (`barang` table) and shipping rates (`ongkir` table). The `init_db()` function ensures that the database is initialized with some sample data.
3. **LangChain**: Used for document retrieval and answering questions. It connects to the vector store (Chroma) to search for relevant documents (product information and shipping details).
4. **Vector Store**: Chroma is used to store the documents, which are indexed and retrieved based on the user's question.
5. **PromptTemplate**: Defines the format in which the question is answered, including how products and shipping rates are displayed.

---

## Troubleshooting:
1. **Model Not Found**: If the Ollama model isn't found, ensure that you've correctly pulled the model using `ollama pull`.
2. **Missing Database**: If you encounter issues with missing tables in the database, ensure that `init_db()` is correctly called during the application setup.
3. **Slow Response**: The first request may take longer because it involves loading the model and vector store. Subsequent requests will be faster.

---

Feel free to modify the code and adapt it for your own use case!

---

**Note**: If you encounter any errors related to missing or incompatible dependencies, ensure that you are using the correct version of Python and all dependencies are installed in your virtual environment (if applicable).
