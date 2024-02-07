import os
import torch
from torch.optim import SGD 
from model import Aam
from general import Config
from dataset import release_dataset, data_Loading

def train():
    pass

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using {}.'.format(device))

    # dataset_path=ROOT_dir+"/dataset"

    # if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
    #     release_dataset(dataset_path)

    # train_path = "{}/train".format(dataset_path)
    # train_ann_path = train_path+"/_annotations.coco.json"
    # valid_path = "{}/valid".format(dataset_path)
    # valid_ann_path = valid_path+"/_annotations.coco.json"

    # config=Config()

    train_path = "E:/mini project/pothole detection/dataset/roboflow/train"
    train_ann_path = "E:/mini project/pothole detection/dataset/roboflow/train/_annotations.coco.json"
    train_data, train_load = data_Loading(train_path, train_ann_path)

    model = Aam().to(device)
    print("Modelparameter", model.parameters())
    optimizer = SGD(model.parameters(),lr=0.001,momentum=0.9)

    epochs = 1
    for t in range(epochs):
        print(f".....Epochs {t+1}.....")    
        model.train()

        for batch, (images, boxes,classes) in enumerate(train_load):
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            boxes_tensor = torch.stack(boxes)   
            category_id = classes.to(device)
            optimizer.zero_grad()
            


if __name__ == "__main__":
    ROOT_dir = os.getcwd()
    main()


    






