#!/bin/sh

# Start Ollama in the background
ollama serve &

# Tunggu sampai Ollama aktif
sleep 10

# Pull model dari Hugging Face jika belum tersedia
ollama pull hf.co/ojisetyawan/gemma2-9b-cpt-sahabatai-v1-instruct-Q4_K_M-GGUF

# Aktifkan virtual environment Python
. /app/venv/bin/activate

# Jalankan Flask
exec python run4.py
