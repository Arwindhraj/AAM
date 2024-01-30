import os
import torch

from general import Config

def train():
    pass

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using Device:",device)
    config=Config()

if __name__ == "__main__":
    main()



