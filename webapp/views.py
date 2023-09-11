from django.shortcuts import render
from django.http import HttpResponse, FileResponse
import torch
from torchvision import transforms
from PIL import Image
import io
from .main import stylize_with_fst

# Create your views here.
def index(request):
    return render(request, 'webapp/index.html')

def stylize(request):
    if request.method == 'POST':
        content_image = Image.open(request.FILES['content_image'])
        style_choice = request.POST['style_choice']
        
        # Map style_choice to the corresponding model path
        style_models = {
            "mosaic": "webapp/saved_models/mosaic.pth",
            "candy": "webapp/saved_models/candy.pth",
            "rain_princess": "webapp/saved_models/rain_princess.pth",
            "udnie": "webapp/saved_models/udnie.pth",
        }
        model_path = style_models.get(style_choice, "path_to_default_model.pth")
        
        # Stylize the image using the chosen model
        output_image = stylize_with_fst(content_image, model_path)
        
        # Convert the PIL image to HttpResponse
        img_byte_array = io.BytesIO()
        output_image.save(img_byte_array, format='PNG')
        response = FileResponse(io.BytesIO(img_byte_array.getvalue()), content_type='image/png')
        
        return response

    else:
        return HttpResponse("Invalid request.")
