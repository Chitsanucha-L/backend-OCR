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
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Path สำหรับ Render

# Helper functions
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def remove_noise(image):
    return cv2.medianBlur(image, 3)

def remove_fine_detail(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

def edge_detection(image, thold1 = 80, thold2 = 110):
    grayscale_image = grayscale(image)
    grayscale_image = remove_fine_detail(grayscale_image)
    grayscale_image = remove_noise(grayscale_image)
    edges = cv2.Canny(grayscale_image, thold1, thold2)
    return edges

def text_region(image, edges):
    og = np.copy(image)
    region_prob = np.zeros_like(edges, dtype='uint8')

    # Detect regions containing text
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # กรอง components ที่เล็กเกินไป
        if w < 20 or h < 20:  
            continue

        crop = og[y:y+h, x:x+w]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_bin = thresholding(crop_gray)

        # แปลง crop เป็น OCR
        rawtext = pytesseract.image_to_string(crop_bin, lang='tha', config='--psm 6')  # PSM 6: Assume a single uniform block of text
        proctext = "".join(rawtext.split())
        
        if len(proctext) > 0:
            region_prob[y:y+h, x:x+w] += 1

    return region_prob

def region_annotate(image, prob, edges):
    og = np.copy(image)
    prob = cv2.GaussianBlur(prob, (15, 15), 15)
    prob = thresholding(prob)

    contours, _ = cv2.findContours(prob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    texts = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # กรอง components ที่เล็กเกินไป
        if w < 20 or h < 20:
            continue

        crop = og[y:y+h, x:x+w]
        region = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        region = thresholding(region)

        rawtext = pytesseract.image_to_string(region, lang='tha', config='--psm 6')  # PSM 6: Assume a single uniform block of text
        proctext = ''.join(rawtext.strip().split('\n'))
        proctext = re.sub('[^ก-๙0-9- ]', '', proctext)

        if len(proctext):
            texts.append(proctext)
            boxes.append((x, y, x+w, y+h))

    # Annotate text on image
    for box, text in zip(boxes, texts):
        x1, y1, x2, y2 = box
        cv2.rectangle(og, (x1, y1), (x2, y2), (0, 255, 0), 2)

        try:
            fontpath = "fonts/ChakraPetch-Bold.ttf"
            font = ImageFont.truetype(fontpath, 32)
        except IOError:
            font = ImageFont.load_default()  # Default font if the custom font is unavailable
            
        img_pil = Image.fromarray(og)
        draw = ImageDraw.Draw(img_pil)
        draw.text((x1, y1),  text, font=font, fill=(128, 255, 0, 0))
        og = np.array(img_pil)

    return og

def process_ocr(image):
    logging.debug("Starting OCR processing")

    edges = edge_detection(image)
    logging.debug("Edge detection completed")

    probs = text_region(image, edges)
    logging.debug("Text region detected")   

    annotated_image = region_annotate(image, probs, edges)
    logging.debug("Region annotation completed")

    return annotated_image

@app.get("/")
def ping():
    return {"status": "ok"}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        np_arr = np.frombuffer(image_data, np.uint8)

        # Decode the image using OpenCV
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Perform OCR and processing
        processed_image = process_ocr(image)

        # Return the result as a response
        _, buffer = cv2.imencode('.png', processed_image)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
    
    except Exception as e:
        logging.error(f"Error processing OCR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
