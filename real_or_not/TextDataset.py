from torch import tensor
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, x_data, y_data= None):
        self.x_data = [tensor(x) for x in x_data]
        self.y_data = y_data

    def __getitem__(self, index):
        if self.y_data is not None:
            return self.x_data[index], self.y_data[index]
        else:
            return self.x_data[index]

    def __len__(self):
        return self.x_data.__len__()
