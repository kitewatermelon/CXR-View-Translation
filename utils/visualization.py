import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


Normalization_Values = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
def DeNormalize(tensor_of_image):
    return tensor_of_image * Normalization_Values[1][0] + Normalization_Values[0][0]

def print_images(image_tensor, num_images, title):
    
    images = DeNormalize(image_tensor)
    images = images.detach().cpu()
    image_grid = make_grid(images[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')  # 축 제거
    if title:
        plt.title(f"{title}")

    plt.show()

def save_image(inputs, generator_image, targets, num_images, i, save_path=None, is_train=True):
    # Ensure num_images doesn't exceed the available batch size
    num_images = min(num_images, inputs.shape[0], generator_image.shape[0], targets.shape[0])

    # De-normalize the images and detach to CPU
    inputs = DeNormalize(inputs).detach().cpu()
    generator_image = DeNormalize(generator_image).detach().cpu()
    targets = DeNormalize(targets).detach().cpu()

    fig, axes = plt.subplots(3, num_images, figsize=(num_images*3, 9))  # 3 rows, num_images columns

    for j in range(num_images):
        # Input images
        axes[0, j].imshow(inputs[j].permute(1, 2, 0).squeeze())
        axes[0, j].axis('off')
        if j == 0:
            axes[0, j].set_title(f"Input data of {i}")

        # Generated images
        axes[1, j].imshow(generator_image[j].permute(1, 2, 0).squeeze())
        axes[1, j].axis('off')
        if j == 0:
            axes[1, j].set_title(f"Generated data of {i}")

        # Target images
        axes[2, j].imshow(targets[j].permute(1, 2, 0).squeeze())
        axes[2, j].axis('off')
        if j == 0:
            axes[2, j].set_title(f"Target data of {i}")

    plt.tight_layout()
    
    string = 'train/' if is_train else 'test/'
    i = 0 
    save_dir = f'{save_path}/img/{string}'
    file_name = f'image_{i}.png'
    while os.path.exists(os.path.join(save_dir, file_name)):
        i += 1
        file_name = f'image_{i}.png'

    # Save the figure
    plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
    plt.close(fig) 

def save_loss(history, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{save_path}/history/history.png')