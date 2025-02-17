import json
import os
import uuid
import logging

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse

from app.config import Configuration
from app.ml.classification_utils import classify_image

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    Handles image uploads, processes them, and classifies the image using the selected model.
    Provides error handling and file cleanup in case of failures.
    """
    try:
        # Ensure the uploaded file is valid
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded.")

        # Read the uploaded file content
        contents = await file.read()

        # Ensure the file is not empty
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Extract file extension or default to .jpg
        extension = os.path.splitext(file.filename)[1].lower()
        if not extension:
            extension = ".jpg"

        # Ensure the extension is valid (optional, based on allowed types)
        allowed_extensions = {".jpg", ".jpeg", ".png"}
        if extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Invalid file type: {extension}. Allowed types: {allowed_extensions}")

        # Generate a unique filename to prevent conflicts
        unique_filename = f"upload_{uuid.uuid4().hex}{extension}"
        file_location = os.path.join(Configuration.image_folder_path, unique_filename)

        # Save the file to disk
        try:
            with open(file_location, "wb") as f:
                f.write(contents)
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save the uploaded file.")

        # Call the existing classification function
        try:
            classification_scores = classify_image(model_id=model_id, img_id=unique_filename)
        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            os.remove(file_location)  # Clean up file if classification fails
            raise HTTPException(status_code=500, detail="Error processing the image.")

        # Return the classification results in the template response
        return request.app.state.templates.TemplateResponse(
            "classification_output.html",
            {
                "request": request,
                "image_id": unique_filename,
                "classification_scores": json.dumps(classification_scores),
            },
        )

    except HTTPException as http_ex:
        # Return an appropriate response for client errors
        return {"error": http_ex.detail}

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": "An unexpected error occurred. Please try again later."}
