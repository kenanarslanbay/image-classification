import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import Configuration
from app.forms.classification_form import ClassificationForm
from app.ml.classification_utils import classify_image
from app.utils import list_images

# NEW: Import the histogram router
from app.routes.histogram import router as histogram_router

app = FastAPI()
config = Configuration()

# Mount static files and configure templates.
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
# Make the templates accessible to all routers (including histogram).
app.state.templates = templates

@app.get("/info")
def info() -> dict[str, list[str]]:
    """Return available models and images."""
    list_of_images = list_images()
    list_of_models = Configuration.models
    return {"models": list_of_models, "images": list_of_images}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/classifications")
def create_classify(request: Request):
    """Render the classification selection page."""
    return templates.TemplateResponse(
        "classification_select.html",
        {"request": request, "images": list_images(), "models": Configuration.models},
    )

@app.post("/classifications")
async def request_classification(request: Request):
    """Handle image classification requests."""
    form = ClassificationForm(request)
    await form.load_data()
    image_id = form.image_id
    model_id = form.model_id
    classification_scores = classify_image(model_id=model_id, img_id=image_id)
    return templates.TemplateResponse(
        "classification_output.html",
        {
            "request": request,
            "image_id": image_id,
            "classification_scores": json.dumps(classification_scores),
        },
    )

# NEW: Include the histogram router.
# The histogram router provides a GET endpoint to display a dropdown for image selection 
# (similar to the classification form) and a POST endpoint to generate and return the histogram.
app.include_router(histogram_router, prefix="/histogram")
