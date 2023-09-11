import torch
import torchvision.transforms as transforms
from PIL import Image
from .transformer_net import TransformerNet  # Assuming you have the repository's code

# Load the pre-trained model
def stylize_with_fst(content_image, model_path):
    model = TransformerNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Image transformations
    preprocess = transforms.Compose([
        transforms.Resize(400),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    postprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.mul(1./255)),
        transforms.ToPILImage()
    ])
    
    # Apply transformations
    content_tensor = preprocess(content_image).unsqueeze(0)
    if torch.cuda.is_available():
        content_tensor = content_tensor.cuda()
    
    # Stylize the image
    with torch.no_grad():
        stylized_tensor = model(content_tensor)
    
    stylized_image = postprocess(stylized_tensor.squeeze(0))
    return stylized_image
