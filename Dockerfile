# -------------------------------
# Base image: Python 3.10 (stable)
# -------------------------------
FROM python:3.10-slim

# -------------------------------
# System dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Set working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# Copy requirements first (cache)
# -------------------------------
COPY requirements.txt .

# -------------------------------
# Install Python dependencies
# -------------------------------
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Copy app code
# -------------------------------
COPY . .

# -------------------------------
# Expose port for Streamlit
# -------------------------------
EXPOSE 8501

# -------------------------------
# Start Streamlit
# -------------------------------
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
