FROM python:3.12-slim

# ติดตั้ง Tesseract และภาษาไทย
RUN apt-get update && \
    apt-get install -y tesseract-ocr tesseract-ocr-tha && \
    rm -rf /var/lib/apt/lists/*

# ติดตั้ง Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดเข้าไปใน container
COPY . .

# สั่งเปิด port 8000
EXPOSE 8000

# สั่งให้ uvicorn รัน FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
