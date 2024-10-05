# PROGRAMMER: Anna Bajszczak
# DATE CREATED: 9/29/2024                                   
# REVISED DATE
# PURPOSE: Predict the class of an image using a trained deep learning model with options
#          to adjust various parameters through command line arguments. The program
#          processes the input image, loads the specified model checkpoint, and outputs
#          the top K predicted classes along with their probabilities. If the user fails 
#          to provide some or all of the inputs, then the default values are used for 
#          the missing inputs. Command Line Arguments:
#     1. Image file for prediction as --image (required)
#     2. Checkpoint file path as --checkpoint (required)
#     3. Number of top predictions to display as --top_k with default value 3
#     4. Category names file as --category_names with default value 'cat_to_name.json'
#     5. Enable GPU for prediction as --gpu with default value 'gpu'


# Imports here
import argparse
import json
import numpy as np
import torch
from torchvision import models
from PIL import Image
import torch.optim as optim
import matplotlib.pyplot as plt

# Function to parse command line arguments
def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, help='path to image file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.', default=3)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
    
    return parser.parse_args()

# Function to load a checkpoint and rebuild the model

def load_checkpoint(path):
    checkpoint = torch.load(path)

    model_name = checkpoint.get('model_name', 'vgg16')
    
    if model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif model_name == 'resnet':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Using ResNet50
    else:
        raise ValueError("Invalid model name in checkpoint.")
    
    for param in model.parameters():
        param.requires_grad = False
        
    # Use get() to avoid KeyError and handle missing keys gracefully
    model.classifier = checkpoint.get('classifier', model.classifier)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epochs']

# Function to process an image
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
    with Image.open(image_path) as im:
        width, height = im.size
        if width < height:
            new_height = 256
            new_width = int((new_height / height) * width)
        else:
            new_width = 256
            new_height = int((new_width / width) * height)

        im_resized = im.resize((new_width, new_height), Image.LANCZOS)

        # Cropping the image to 224x224
        left = (new_width - 224) / 2
        top = (new_height - 224) / 2
        right = (new_width + 224) / 2
        bottom = (new_height + 224) / 2

        im_cropped = im_resized.crop((left, top, right, bottom))
        
        # Converting to numpy array and normalize
        np_image = np.array(im_cropped) / 255
        np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        return np_image.transpose((2, 0, 1))

# Function to predict the class of an image
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu == "gpu" else 'cpu')
    image = process_image(image_path)  # Processing image
    
    model.to(device)  # Move model to device (CPU or GPU)
    model.eval()  # Evaluation mode
    
    with torch.no_grad():  # Disable gradients for efficiency
        image_tensor = torch.from_numpy(image).float()  # Converting image to tensor
        image_tensor = image_tensor.unsqueeze(0)  # Adding batch dimension
        output = model(image_tensor.to(device))  # Move image_tensor to the same device as the model
        probabilities = torch.softmax(output, dim=1)  # Converting output to probabilities using softmax
        
        top_probs, top_indices = probabilities.topk(topk)  # Retrieve top K probabilities and their indices
    
        # Converting probabilities to numpy arrays
        top_probs = top_probs.cpu().numpy().flatten()  # Move to CPU for numpy conversion
        top_indices = top_indices.cpu().numpy().flatten()
    
        # Mapping indices to classes
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}  # Invert class_to_idx
        top_classes = [idx_to_class[idx] for idx in top_indices]

    return top_probs, top_classes

# Main function
if __name__ == "__main__":
    args = arg_parser()  # Parse command-line arguments

    # Load the model architecture and checkpoint
    model, optimizer, epochs = load_checkpoint(args.checkpoint)  # Passing checkpoint path from args

    # Call the predict function
    top_probs, top_classes = predict(args.image, model, args.top_k)  # Passing image path and top_k from args

    # Load category names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Convert class indices to flower names
    top_flower_names = [cat_to_name[str(cls)] for cls in top_classes]

    # Print probabilities and flower names
    print("Top Probabilities:", top_probs)
    print("Top Flower Names:", top_flower_names)

    # Displaying the image
    image = process_image(args.image)  # Process the image to display
    image_display = np.clip(image.transpose((1, 2, 0)), 0, 1)  # Scale to [0, 1]
    plt.imshow(image_display)  # Display image in the correct format
    plt.title("Predicted Flower")
    plt.axis('off')
    plt.show()

    # Creating a bar graph for the probabilities
    plt.figure(figsize=(10, 6))  # Create a new figure for the bar graph
    y_pos = np.arange(len(top_flower_names))
    plt.barh(y_pos, top_probs, align='center', height=0.4)
    plt.yticks(y_pos, top_flower_names)
    plt.xlabel('Probability')
    plt.title('Top Flower Predictions')
    plt.show()

