import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import torchvision

# Functon to plot story
# function to plot metrics
# Function to plot images and labels
class FaceDataset(Dataset):

    def __init__(self, image_dir, transform=None):
        
        """Function to load images into Tensor
            Args: 
                - image_dir : directory of images
                - Return : a dictonary with images and labels
                """
        self.image_dir = image_dir
        self.image_dict = self.load_image()
        self.transform = transform


    def __len__(self) :
        return len(self.image_dict["label"])


    def __getitem__(self, index) :
        
        
        path = torch.from_numpy(io.imread(self.image_dict["img_dir"][index],
                                         as_gray=True).astype(np.float32)).unsqueeze(0)
        label = self.image_dict["label_bin"][index]

        if self.transform:
            path = self.transform(path)
        
        return path, label


    def load_image(self) :
        img_dict = {"img_dir" : [], "label" : [], 'label_bin':[]}
        for root, dirs, files in os.walk(self.image_dir):
            for img in files:
                img_dict["img_dir"].append(os.path.join(root, img))
                
                img_dict["label"].append(img[:4])
                if img[:4] != 'real':
                    img_dict["label_bin"].append(0.0)
                else:
                    img_dict["label_bin"].append(1.0)
        img_dict["label_bin"] = torch.tensor(img_dict["label_bin"],dtype=torch.float32)
        return img_dict


def plot_history(history, figsize=(8,6), 
                 plot={"Accuracy":['accuracy','val_accuracy'], 'Loss':['loss', 'val_loss']},
                 save_as='auto'):
    """
    Show history
    args:
        history: history
        figsize: fig size
        plot: list of data to plot : {<title>:[<metrics>,...], ...}
    """
    fig_id=0
    for title,curves in plot.items():
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.ylabel(title)
        plt.xlabel('Epoch')
        for c in curves:
            plt.plot(history.history[c])
        plt.legend(curves, loc='upper left')
        plt.show()

def plot_images():
    pass