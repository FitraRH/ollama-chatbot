# Gunakan Debian Slim untuk ukuran yang lebih kecil
FROM debian:bookworm-slim

# Set environment variable agar proses instalasi non-interaktif
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies yang paling minimal
RUN apt update && apt install -y \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama (AI Model Server)
RUN curl -fsSL https://ollama.com/install.sh | bash

# Buat direktori kerja
WORKDIR /app

# Copy dependencies Python
COPY requirements.txt requirements.txt

# Install Python dependencies
# Gunakan Virtual Environment untuk Python
RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy semua file aplikasi
COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Jalankan Flask saat container dimulai
#CMD ["/app/venv/bin/python", "run4.py"]
