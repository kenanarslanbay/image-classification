import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import Configuration
from app.forms.classification_form import ClassificationForm
from app.ml.classification_utils import classify_image
from app.utils import list_images

# New import for the upload routes.
from app.routes.upload import router as upload_router
# New import for the histogram routes.
from app.routes.histogram import router as histogram_router

app = FastAPI()

# Mount static files and configure templates.
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
# Make the templates available in the app state so submodules (upload, histogram) can use them.
app.state.templates = templates

# Instantiate configuration
config = Configuration()

@app.get("/info")
def info() -> dict[str, list[str]]:
    list_of_images = list_images()
    list_of_models = Configuration.models
    return {"models": list_of_models, "images": list_of_images}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/classifications")
def create_classify(request: Request):
    return templates.TemplateResponse(
        "classification_select.html",
        {"request": request, "images": list_images(), "models": Configuration.models},
    )

@app.post("/classifications")
async def request_classification(request: Request):
    form = ClassificationForm(request)
    await form.load_data()
    if not form.is_valid():
        return templates.TemplateResponse(
            "classification_select.html",
            {"request": request, "errors": form.errors, "images": list_images(), "models": Configuration.models},
        )
    image_id = form.image_id
    model_id = form.model_id
    classification_scores = classify_image(model_id=model_id, img_id=image_id)
    return templates.TemplateResponse(
        "classification_output.html",
        {"request": request, "image_id": image_id, "classification_scores": json.dumps(classification_scores)},
    )

# Include upload routes under the /classifications prefix.
app.include_router(upload_router, prefix="/classifications")
# Include histogram routes under the /histogram prefix.
app.include_router(histogram_router, prefix="/histogram")
