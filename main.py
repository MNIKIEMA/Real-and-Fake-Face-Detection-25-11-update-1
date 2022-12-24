import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import FaceDataset
if __name__ == "__main__":
    data = FaceDataset("data")
    """plt.figure()
    plt.imshow(data[2][0])
    plt.show()"""
    data_loader = DataLoader(dataset= data,batch_size=10, shuffle=True)
    print(next(iter(data_loader))[0].shape)