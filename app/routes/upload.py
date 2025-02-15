

import json
import os
import uuid

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse

from app.config import Configuration
from app.ml.classification_utils import classify_image

router = APIRouter()

@router.get("/upload", response_class=HTMLResponse)
def upload_classification_form(request: Request):
    """
    Returns a form that allows the user to upload an image file
    and select a model for classification.
    """
    return request.app.state.templates.TemplateResponse(
        "classification_upload.html",
        {"request": request, "models": Configuration.models},
    )

@router.post("/upload")
async def classify_uploaded_image(
    request: Request,
    file: UploadFile = File(...),
    model_id: str = Form(...)
):
    """
    Saves the uploaded image in the image folder and passes its filename
    to the existing classification API. The output page shows the uploaded image
    along with the classification scores.
    """
    # Read the uploaded file content.
    contents = await file.read()

    # Use the original file extension if available; otherwise, default to .jpg.
    extension = os.path.splitext(file.filename)[1]
    if not extension:
        extension = ".jpg"

    # Generate a unique filename so that uploaded files do not conflict.
    unique_filename = f"upload_{uuid.uuid4().hex}{extension}"
    file_location = os.path.join(Configuration.image_folder_path, unique_filename)
    
    # Save the file.
    with open(file_location, "wb") as f:
        f.write(contents)
    
    # Call the existing classify_image function (which uses the file by name).
    classification_scores = classify_image(model_id=model_id, img_id=unique_filename)
    
    # Return the output template (reuse classification_output.html) and
    # pass the uploaded image’s filename so that it can be displayed.
    return request.app.state.templates.TemplateResponse(
        "classification_output.html",
        {
            "request": request,
            "image_id": unique_filename,
            "classification_scores": json.dumps(classification_scores),
        },
    )
