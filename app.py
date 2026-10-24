#open the terminal and run this command to install all dependencies
#pip install fastapi 
# pip install uvicorn 
# pip install jinja2 
# pip install python-multipart 
# pip install requests
#pip install uuid
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import uuid
import os

app = FastAPI(title="Clipdrop Image Generator API")

# Static & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# CONFIG
CLIPDROP_API_KEY = "--------"
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/generate-image", response_class=HTMLResponse)
def generate_image(request: Request, prompt: str = Form(...)):

    url = "https://clipdrop-api.co/text-to-image/v1"
    headers = {"x-api-key": CLIPDROP_API_KEY}
    files = {"prompt": (None, prompt)}

    response = requests.post(url, headers=headers, files=files)

    if response.status_code != 200:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Image generation failed"}
        )

    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(response.content)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "prompt": prompt,
            "image_name": filename
        }
    )


@app.get("/download/{image_name}")
def download_image(image_name: str):
    file_path = os.path.join(OUTPUT_DIR, image_name)
    return FileResponse(file_path, media_type="image/png")
