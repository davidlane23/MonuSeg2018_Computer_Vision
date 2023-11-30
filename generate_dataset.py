import os
from modules.MonuSeg_dataset import MonuSegDataset, generate_datasets


def run(data_path):
    # Generate datasets for training, testing and val for repoducibility
    save_dir = "MonuSeg_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading dataset...")
    # if(not os.listdir(data_path)):

    generate_datasets(data_path, save_dir)


if __name__ == "__main__":
    run("data")
