import os
import cv2
import json
import torch
import zipfile
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import FloatTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class DataLoading(Dataset):

    def __init__(self,image,annotation,transform):
        
        """
        Initialize the DataLoading.

        Args:
        - image (str): Path to the image directory.
        - annotation (str): Path to the annotation file (in JSON format).
        - transform (callable): A function or transform to apply to the images.
        """

        self.image = image
        annotation_file = annotation
        self.annotation = json.load(open(annotation_file))
        self.transform = transform

    def __len__(self):

        """
        Returns the number of images in the dataset.
        """

        return len(self.annotation['images'])
        
    def __getitem__(self, idx):

        """
        Retrieves an item (image, bounding box, category ID) from the dataset.

        Args:
        - idx    : Index of the item to retrieve which iterates.

        Returns:
        - image  : Transformed image tensor.
        - bbox   : Bounding box coordinates as a FloatTensor.
        - cat_id : Category ID.
        """

        image_path = os.path.join(self.image,self.annotation['images'][idx]['file_name'])
        image = Image.open(image_path) 

        image = self.transform(image)
        bbox = FloatTensor(self.annotation['annotations'][idx]['bbox'])
        class_id = self.annotation['annotations'][idx]['category_id']

        return image, bbox, class_id

    def collate_fn(self, batch):

        """
        Custom collate function for handling data in batches.

        Args:
        - batch   : List of items, where each item is (image, bbox, cat_id).

        Returns:
        - images  : Batched images.
        - boxes   : Batched bounding boxes.
        - classes : Batched category IDs.
        """

        images = list()
        boxes = list()
        cl_id = list()

        for b in batch:
            image, box, id = b 
            images.append(image)
            boxes.append(box)
            cl_id.append(id)

        images = torch.stack(images,dim=0)
        boxes = torch.stack(boxes,dim=0)
        classes = torch.tensor(cl_id, dtype=torch.long)

        return images,boxes,classes
    
def compute_mean_std(dataset_path):

    """
    Compute the mean and standard deviation of the dataset.

    Args:
    - dataset_path (str): The path to the dataset.

    Returns:
    - mean : Mean values for each channel.
    - std  : Standard deviation values for each channel.
    """

    mean = np.zeros(3)
    std = np.zeros(3)
    num_images = 0

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg')):

                image_path = os.path.join(root, file)
                img = cv2.imread(image_path) / 255.0  

                mean += np.mean(img, axis=(0, 1))
                std += np.std(img, axis=(0, 1))

                num_images += 1

    mean /= num_images
    std /= num_images

    return mean, std

def data_Loading(train_path,train_ann_path):

    """
    Load the training data in DataLoader.

    Args:
    - train_path     : Path to the training image directory.
    - train_ann_path : Path to the training annotation file (in JSON format).

    Returns:
    - train_data (DataLoading): Custom dataset for training data.
    - train_load (DataLoader): DataLoader for training data.
    """

    input_size = (256,256)

    train_path = train_path

    train_ann_path = train_ann_path

    mean,std = compute_mean_std(train_path)

    transform = transforms.Compose([transforms.Resize(input_size),transforms.ToTensor(),transforms.Normalize(mean=mean,std=std),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ColorJitter(),transforms.RandomRotation(degrees=90),transforms.RandomGrayscale(),transforms.GaussianBlur(kernel_size=[3,3]),transforms.RandomAdjustSharpness(sharpness_factor=0.5),transforms.RandomAutocontrast()])

    train_data = DataLoading(image=train_path,annotation=train_ann_path,transform=transform)

    train_load = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=train_data.collate_fn, num_workers=4, pin_memory=True)

    print("Number of batches loaded:", len(train_load))

    return train_data, train_load

class release_dataset():
    def __init__(dataset_dir):
        url = "https://github.com/Arwindhraj/custom_nn/releases/download/Dataset/Pothole.v3i.coco.zip"  
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))

        block_size = 1024  
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True,desc="Downloading dataset : ")

        with open("dataset.zip", "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("Downloading Dataset Failed")
        
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)