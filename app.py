from fastapi import FastAPI, File, UploadFile
from PIL import Image
import pytesseract
import io
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ตั้งค่าพาธของ Tesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Path สำหรับ Render

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/ocr/")
async def ocr(file: UploadFile = File(...)):
    try:
        # อ่านไฟล์ภาพจาก request
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # ทำ OCR โดยใช้ pytesseract
        text = pytesseract.image_to_string(image, lang='tha')

        return JSONResponse(content={"text": text})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
