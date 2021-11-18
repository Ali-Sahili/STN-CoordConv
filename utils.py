import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.
def visualize_stn(model, test_loader, device, enable_coordconv=False):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        if enable_coordconv:
            data = model.coordconv(data)
            
        transformed_input_tensor = model.stn(data).cpu()
        if transformed_input_tensor.shape[2] > 1:
            transformed_input_tensor = transformed_input_tensor[:,1].view(-1,1,28,28)
        print(input_tensor.shape, transformed_input_tensor.shape)
        
        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))
        out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')
    
    plt.show()
