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

def red_channel(image):
    return image[:, :, 2]

def green_channel(image):
    return image[:, :, 1]

def blue_channel(image):
    return image[:, :, 0]

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def remove_noise(image):
    return cv2.medianBlur(image, 3)

def remove_fine_detail(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

def edge_detection(image, thold1 = 80, thold2 = 110):
    og = np.copy(image)
    r_ch = red_channel(og)
    g_ch = green_channel(og)
    b_ch = blue_channel(og)
    l_ch = grayscale(og)
    r_proc = remove_fine_detail(r_ch)
    g_proc = remove_fine_detail(g_ch)
    b_proc = remove_fine_detail(b_ch)
    l_proc = remove_fine_detail(l_ch)
    r_proc = remove_noise(r_proc)
    g_proc = remove_noise(g_proc)
    b_proc = remove_noise(b_proc)
    l_proc = remove_noise(l_proc)
    r_edge = cv2.Canny(r_proc, thold1, thold2)
    g_edge = cv2.Canny(g_proc, thold1, thold2)
    b_edge = cv2.Canny(b_proc, thold1, thold2)
    l_edge = cv2.Canny(l_proc, thold1, thold2)
    edges = cv2.bitwise_or(r_edge, g_edge)
    edges = cv2.bitwise_or(edges, b_edge)
    edges = cv2.bitwise_or(edges, l_edge)
    cross_kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cross_kernel3)
    edges = cv2.dilate(edges, cross_kernel3)
    rect_kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, rect_kernel3)
    # rect_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # edges = cv2.erode(edges, rect_kernel2)
    ellipse_kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mid = cv2.erode(edges, ellipse_kernel5)
    edges = cv2.subtract(edges, mid)
    return edges

def filtered_component(image, edges, area_cutoff = 15, width_cutoff = 30.0, height_cutoff = 50.0, color_cutoff = 30.0, ver_align_cutoff = 10, dis_cutoff = 10.0):
    og = np.copy(image)
    comp = cv2.bitwise_not(edges)
    img_w = image.shape[1]
    img_h = image.shape[0]
    img_size = img_w * img_h
    analysis = cv2.connectedComponentsWithStats(comp, 4, cv2.CV_32S)
    (totalLabels, label_ids, stats, centroid) = analysis

    output = np.zeros(edges.shape, dtype="uint8")

    for i in range(1, totalLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        if 10 < area < img_size/area_cutoff and \
            5 < width < (width_cutoff*img_w/100.0) and \
            5 <height < (height_cutoff*img_h/100.0):
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)

    comp_mask1 = output
    
    analysis = cv2.connectedComponentsWithStats(comp_mask1,
                                                4,
                                                cv2.CV_32S)
    (totalLabels, label_ids, stats, centroid) = analysis

    output = np.zeros(edges.shape, dtype="uint8")
    used = []
    parent = []
    child = []
    mean_color = []
    comp_coord = {}
    comp_dist = {}

    for i in range(1, totalLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        top = stats[i, cv2.CC_STAT_TOP]
        left = stats[i, cv2.CC_STAT_LEFT]

        rgb = np.array([0,0,0])

        coord = []

        for row in range(top, top+height+1):
            for col in range(left, left+width+1):
                if row < img_h and col < img_w:
                    if label_ids[row, col] == i:
                        rgb += og[row, col]
                        coord.append((row, col))
        comp_coord[i] = coord

        rgb = rgb/area
        mean_color.append(rgb)

    for i in range(1, totalLabels-1):
        iw = stats[i, cv2.CC_STAT_WIDTH]
        ih = stats[i, cv2.CC_STAT_HEIGHT]
        it = stats[i, cv2.CC_STAT_TOP]
        il = stats[i, cv2.CC_STAT_LEFT]
        ib = it + ih
        ir = il + iw
        for j in range(i+1, totalLabels):
            jw = stats[j, cv2.CC_STAT_WIDTH]
            jh = stats[j, cv2.CC_STAT_HEIGHT]
            jt = stats[j, cv2.CC_STAT_TOP]
            jl = stats[j, cv2.CC_STAT_LEFT]
            jb = jt + jh
            jr = jl + jw

            hor = jl - ir
            if hor < 0:
                hor = 1
            tt = np.abs(it - jt)
            bb = np.abs(ib - jb)
            if tt < ver_align_cutoff or bb < ver_align_cutoff:
                comp_dist[(i, j)] = hor
            else:
                comp_dist[(i, j)] = 200

    for i in range(1, totalLabels-1):
        child = [i]
        if i not in used:
            output = cv2.bitwise_or(output, (label_ids == i).astype("uint8") * (len(parent) + 2))
            for j in range(i+1, totalLabels):
                if j in used:
                    continue
                if np.linalg.norm(mean_color[i-1]-mean_color[j-1]) < color_cutoff:
                    close_enough = False
                    for k in child:
                        if comp_dist[(k, j)] < dis_cutoff:
                            close_enough = True
                            break
                    if close_enough:
                        used.append(j)
                        child.append(j)
                        if i not in used:
                            used.append(i)
                        if i not in parent:
                            parent.append(i)
                        output = cv2.bitwise_or(output, (label_ids == j).astype("uint8") * (len(parent) + 1))
    
    components = output
    return components

def text_region(image, components):
    og = np.copy(image)
    region_prob = np.zeros_like(components, dtype='uint8')

    for comp_idx in np.unique(components):
        if comp_idx == 0:
            continue

        comp = (components == comp_idx).astype('uint8')*255
        l, r, t, b = 0,0,0,0
        for row in range(og.shape[0]):
            if (np.sum(comp[row, :]) > 0):
                if t == 0:
                    t = row
                b = row
        for col in range(og.shape[1]):
            if (np.sum(comp[:, col]) > 0):
                if l == 0:
                    l = col
                r = col
        crop = og[t:b, l:r]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = thresholding(crop)

        ocr_img = np.zeros_like(components, dtype="uint8")
        for row in range(t, b):
            for col in range(l, r):
                ocr_img[row, col] = crop[row-t, col-l]
        rawtext = pytesseract.image_to_string(ocr_img, lang='tha')
        proctext = "".join(rawtext.split())
        if len(proctext) > 0:
            for row in range(t, b):
                for col in range(l, r):
                    region_prob[row, col] += 1

    return region_prob

def region_annotate(image, prob, comp, padding_size = 50):
    og = np.copy(image)
    mask = (comp > 0).astype('uint8')*255
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    masked = cv2.bitwise_and(og, og, mask)
    boxes = []
    texts = []

    # prob[prob > 1] = prob[prob > 1] - 1
    prob = cv2.GaussianBlur(prob, (15, 15), 15)
    prob = cv2.GaussianBlur(prob, (15, 15), 15)
    # prob[prob > 1] = prob[prob > 1] - 1
    prob = thresholding(prob)

    analysis = cv2.connectedComponentsWithStats(prob,
                                                4,
                                                cv2.CV_32S)
    (totalLabels, label_ids, stats, centroid) = analysis

    for i in range(1, totalLabels):
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        mask = (label_ids == i).astype("uint8") * 255
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5)))

        top = max(stats[i, cv2.CC_STAT_TOP] - 10, 3)
        left = max(stats[i, cv2.CC_STAT_LEFT] - 20, 3)
        bottom = min(top + height + 10, og.shape[0] - 4)
        right = min(left + width + 100, og.shape[1] - 4)


        crop = og[top:bottom, left:right]
        # plt.imshow(crop)
        # plt.show()
        # region = crop
        region = grayscale(crop)
        l = grayscale(crop)
        r = red_channel(crop)
        g = green_channel(crop)
        b = blue_channel(crop)
        base_var = 0
        best = -1
        grads = [l, r, g, b]
        for n, temp in enumerate(grads):
            temp_var = np.var(temp) 
            if temp_var > base_var:
                best = n
                base_var = temp_var
        region = grads[best]
        region = cv2.blur(region,(3,3))
        region = thresholding(region)
        # region = np.pad(region, padding_size, mode='constant', constant_values=0)
        temp = np.zeros_like(prob)
        temp[top:bottom, left:right] = region
        region = temp

        rawtext = pytesseract.image_to_string(region, lang='tha')
        proctext = ''.join(rawtext.strip().split('\n'))
        re.sub('[^ก-๙0-9- ]', '', proctext)
        if len(proctext):
            texts.append(proctext)
            boxes.append((top, left, bottom, right))

        # print(pytesseract.image_to_boxes(region, lang='tha'))

    for box, text in zip(boxes, texts):
        top, left, bottom, right = box
        cv2.rectangle(og, (left, top), (right, bottom), (0, 255, 0), 2)

        fontpath = "fonts/ChakraPetch-Bold.ttf"
        font = ImageFont.truetype(fontpath, 32)
        img_pil = Image.fromarray(og)
        draw = ImageDraw.Draw(img_pil)
        draw.text((left, top),  text, font = font, fill = (128, 255, 0, 0))
        og = np.array(img_pil)
    annotated = og
    return annotated

def process_ocr(image, suffix=''):
    logging.debug("Starting OCR processing")
    edges = edge_detection(image)
    logging.debug("Edge detection completed")
    cv2.imwrite(f'edges{suffix}.tif', edges)
    comps = filtered_component(image, edges)
    logging.debug("Filtered component completed")
    cv2.imwrite(f'comps{suffix}.tif', cv2.equalizeHist(comps))
    probs = text_region(image, comps)
    logging.debug("Text region detected")
    cv2.imwrite(f'probs{suffix}.tif', cv2.equalizeHist(probs))
    annotated = region_annotate(image, probs, comps)
    logging.debug("Region annotation completed")
    return annotated

@app.get("/")
def ping():
    return {"status": "ok"}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        np_arr = np.frombuffer(image_data, np.uint8)

        # Step 3: Decode the image using OpenCV
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        suffix = "_debug"

        # Perform OCR and processing
        processed_image = process_ocr(image, suffix=suffix)

        # Step 5: Save the image with bounding boxes to disk (debug only)
        cv2.imwrite(f"annotated{suffix}.png", processed_image)

        # Step 6: Return the result as a response
        _, buffer = cv2.imencode('.png', processed_image)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
