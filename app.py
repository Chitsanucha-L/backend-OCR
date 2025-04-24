from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageFont, ImageDraw
import pytesseract
import io
import cv2
import numpy as np
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import re

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
)

# ตั้งค่าพาธของ Tesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Helper functions
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def remove_noise(image):
    return cv2.medianBlur(image, 3)

def remove_fine_detail(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

def edge_detection(image, thold1=80, thold2=110):
    grayscale_image = grayscale(image)
    grayscale_image = remove_fine_detail(grayscale_image)
    grayscale_image = remove_noise(grayscale_image)
    edges = cv2.Canny(grayscale_image, thold1, thold2)
    return edges

def text_regions_combined(image, edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 20 or h < 20:
            continue
        boxes.append((x, y, x + w, y + h))

    return boxes

def region_annotate(image, boxes):
    og = np.copy(image)
    texts = []

    for x1, y1, x2, y2 in boxes:
        crop = og[y1:y2, x1:x2]
        region = thresholding(grayscale(crop))

        rawtext = pytesseract.image_to_string(region, lang='tha', config='--psm 6')
        proctext = re.sub('[^ก-๙0-9- ]', '', rawtext.replace('\n', ''))

        if len(proctext.strip()) == 0:
            continue

        texts.append(proctext)
        cv2.rectangle(og, (x1, y1), (x2, y2), (0, 255, 0), 2)

        try:
            font = ImageFont.truetype("fonts/ChakraPetch-Bold.ttf", 32)
        except IOError:
            font = ImageFont.load_default()

        img_pil = Image.fromarray(og)
        draw = ImageDraw.Draw(img_pil)
        draw.text((x1, y1), proctext, font=font, fill=(128, 255, 0, 0))
        og = np.array(img_pil)

    return og

def process_ocr(image):
    logging.debug("Starting OCR processing")
    edges = edge_detection(image)
    logging.debug("Edge detection completed")

    boxes = text_regions_combined(image, edges)
    logging.debug(f"Found {len(boxes)} text regions")

    annotated_image = region_annotate(image, boxes)
    logging.debug("Annotation completed")

    return annotated_image

@app.get("/")
def ping():
    return {"status": "ok"}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        processed_image = process_ocr(image)

        _, buffer = cv2.imencode('.png', processed_image)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

    except Exception as e:
        logging.error(f"Error processing OCR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)