from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import re
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def clean_text(text):
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', '', text)
    return text.strip()

def simple_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    config = "--oem 1 --psm 6"
    data = pytesseract.image_to_data(thresh, lang="tha", config=config, output_type=pytesseract.Output.DICT)

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("fonts/ChakraPetch-Bold.ttf", 28)
    except:
        font = ImageFont.load_default()

    n_boxes = len(data['level'])
    for i in range(n_boxes):
        text = clean_text(data['text'][i])
        if text:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            draw.rectangle([x, y, x + w, y + h], outline="green", width=2)
            draw.text((x, y - 25), text, font=font, fill=(0, 255, 0))

    return np.array(img_pil)

@app.get("/")
def ping():
    return {"status": "ok"}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        processed_image = simple_ocr(image)

        _, buffer = cv2.imencode('.png', processed_image)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
    
    except Exception as e:
        logging.error(f"OCR Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
