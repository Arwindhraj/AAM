import os
import torch

from general import Config
from dataset import release_dataset

def train():
    pass

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using {}.'.format(device))

    dataset_path=ROOT_dir+"/dataset"

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        release_dataset(dataset_path)

    train_path = "{}/train".format(dataset_path)
    train_ann_path = train_path+"/_annotations.coco.json"
    valid_path = "{}/valid".format(dataset_path)
    valid_ann_path = valid_path+"/_annotations.coco.json"

    config=Config()

if __name__ == "__main__":
    ROOT_dir = os.getcwd()
    main()


    






