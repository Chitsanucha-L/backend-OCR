from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import pytesseract
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_imagefile(file) -> np.ndarray:
    image = Image.open(io.BytesIO(file))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def resize_image(image, max_width=1000):
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)  # เบากว่า gaussian
    return blur

def extract_text_regions(image):
    # OCR พร้อมตำแหน่งกล่องข้อความ
    data = pytesseract.image_to_data(image, lang='tha', output_type=pytesseract.Output.DICT)
    boxes = []
    texts = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30 and data['text'][i].strip():
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            boxes.append([x, y, x + w, y + h])
            texts.append(data['text'][i])
    return boxes, texts

@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    try:
        image = read_imagefile(await file.read())
        image = resize_image(image)
        preprocessed = preprocess_image(image)
        boxes, texts = extract_text_regions(preprocessed)
        return JSONResponse(content={"boxes": boxes, "texts": texts})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
