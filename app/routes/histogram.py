import io
import base64
import os
import matplotlib.pyplot as plt
import numpy as np
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from PIL import Image

from app.config import Configuration
from app.utils import list_images

# Declare the router.
router = APIRouter()

def generate_histogram(image_path: str) -> str:
    """
    Generate a simple grayscale histogram for the given image.
    The image is converted to grayscale and a filled histogram is plotted.
    Returns the plot as a base64 encoded PNG image.
    """
    # Open image in grayscale.
    img = Image.open(image_path).convert("L")
    np_img = np.array(img)
    # Compute the histogram with 256 bins.
    hist, bins = np.histogram(np_img.flatten(), bins=256, range=[0, 256])
    
    # Create the plot.
    plt.figure(figsize=(8, 4))
    plt.plot(bins[:-1], hist, color='blue', linewidth=2)
    plt.fill_between(bins[:-1], hist, color='lightblue', alpha=0.5)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title("Image Histogram")
    plt.tight_layout()
    
    # Save the plot to a buffer.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Return the image as base64.
    return base64.b64encode(buf.read()).decode("utf-8")

@router.get("/select", response_class=HTMLResponse)
def get_histogram_form(request: Request):
    """
    Render a form for selecting an image to generate a histogram.
    """
    return request.app.state.templates.TemplateResponse(
        "histogram_select.html",
        {"request": request, "images": list_images()}
    )

@router.post("/select", response_class=HTMLResponse)
async def show_histogram(request: Request, image_id: str = Form(...)):
    """
    Process the selected image and display its histogram.
    """
    image_path = os.path.join(Configuration().image_folder_path, image_id)
    histogram_img = generate_histogram(image_path)
    return request.app.state.templates.TemplateResponse(
        "histogram_output.html",
        {"request": request, "image_id": image_id, "histogram_img": histogram_img}
    )
