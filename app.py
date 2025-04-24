from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
import numpy as np
import cv2
import io
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

# ตั้งค่า path ของ Tesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Helper functions

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary

def detect_text_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 30 and h > 20:
            boxes.append((x, y, x+w, y+h))
    return boxes

def annotate_image(image, boxes):
    config = "--oem 1 --psm 11"
    for (x1, y1, x2, y2) in boxes:
        roi = image[y1:y2, x1:x2]
        roi_pre = preprocess_image(roi)
        text = pytesseract.image_to_string(roi_pre, lang="tha", config=config)
        text = ''.join(text.strip().splitlines())
        if text:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return image

@app.get("/")
def ping():
    return {"status": "ok"}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        boxes = detect_text_boxes(image)
        result_image = annotate_image(image, boxes)

        _, buffer = cv2.imencode('.png', result_image)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

    except Exception as e:
        logging.error(f"OCR Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)