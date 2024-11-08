import torch
<<<<<<< HEAD

def collate_fn(batch):
    ids = [item['id'] for item in batch]
    images = [item['image'] for item in batch]
    captions = [item['caption'] for item in batch]
    labels = [item['label'] for item in batch]
    
    return ids, images, captions, labels
=======
from PIL import Image
import torch.nn.functional as F

def pad_image(image, target_height, target_width):
    _, height, width = image.shape
    pad_height = target_height - height
    pad_width = target_width - width
    
    padding = (
        pad_width // 2,                     # Padding bên trái
        pad_width - pad_width // 2,         # Padding bên phải
        pad_height // 2,                    # Padding phía trên
        pad_height - pad_height // 2        # Padding phía dưới
    )
    
    padded_image = F.pad(image, padding, value=0)
    return padded_image

def find_max_size(img_tensors):
    max_height = max(tensor.shape[1] for tensor in img_tensors)  # Lấy chiều cao lớn nhất
    max_width = max(tensor.shape[2] for tensor in img_tensors)   # Lấy chiều rộng lớn nhất
    return max_height, max_width

def preprocessing_label(label: str):
    if label is None:
        return None
    label_mapping = {'not-sarcasm': 0, 'multi-sarcasm': 1, 'image-sarcasm': 2, 'text-sarcasm': 3}
    length_labels = len(label_mapping)
    new_label = torch.zeros(length_labels)
    new_label[label_mapping[label]] = 1
    return new_label

def collate_fn(batch): 
    ids = [item['id'] for item in batch]
    images = [torch.tensor(item['image']) for item in batch]
    captions = [item['caption'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Convert images to PyTorch tensors and permute dimensions from (H, W, C) to (C, H, W)
    images_tensors = [image.permute(2, 0, 1).float() for image in images]

    # Find the maximum height and width among the images
    max_height, max_width = find_max_size(images_tensors)

    # Pad the images to have the same size
    # padded_images = []
    # for img_tensor in images_tensors:
    #     padding = (0, max_width - img_tensor.shape[2], 0, max_height - img_tensor.shape[1])  # (left, right, top, bottom)
    #     padded_image = torch.nn.functional.pad(img_tensor, padding, value=0)  # Pad with 0s (black)
    #     padded_images.append(padded_image)
    padded_images = [pad_image(image, max_height, max_width) for image in images_tensors]

    return captions, padded_images, labels, ids
>>>>>>> 7eb23a3c720d02f8ed4357e6bcb110787aa71cce
