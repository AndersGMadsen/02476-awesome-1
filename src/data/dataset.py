import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import join, exists

class BCSSDataset(Dataset):
    
    def __init__(self, root_dir, key):
        assert key in ["train", "test", "validation"], "Invalid argument"
        
        self.dir = join(root_dir, key)

        assert exists(self.dir), "Dataset not found"

        self.names = sorted([name for name in listdir(self.dir) if name.endswith('.pt')])
        
    def __len__(self,):
        return len(self.names)
    
    def __getitem__(self, idx):
        
        try:
            image, mask = torch.load(join(self.dir, self.names[idx]))
        except Exception as error:
            print("Error:", error)
            print(idx)

        image = image.to(torch.float) / 255.0
        mask = mask.to(torch.float)        
        return image,mask