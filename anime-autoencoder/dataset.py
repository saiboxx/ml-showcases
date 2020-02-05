import os
import pickle
from tqdm import tqdm
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import \
    Compose, ToPILImage, Resize, CenterCrop, ToTensor, \
    RandomPerspective, RandomRotation, RandomResizedCrop


class AnimeFaceDataset(Dataset):
    """
    Data is available at https://github.com/Mckinsey666/Anime-Face-Dataset
    """

    def __init__(self, directory: str):
        self.directory = directory
        self.transform_load = Compose([ToPILImage(), Resize(64), CenterCrop(64), ToTensor()])
        self.images = self.get_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> list:
        return self.images[idx]

    def get_images(self) -> list:
        print("Loading dataset to memory.")
        files = os.listdir(self.directory)
        if "data.pkl" in files:
            images = pickle.load(open(os.path.join(self.directory, "data.pkl"), "rb"))
            print("Loaded from pickle")

        else:
            images = []
            for f in tqdm(files):
                path = os.path.join(self.directory, f)
                try:
                    images.append(io.imread(path))
                except:
                    pass
            pickle.dump(images, open(os.path.join(self.directory, "data.pkl"), "wb"))
            print("Saved images to pickle file.")

        print("Dataset consists of {} images".format(len(images)))
        return [self.transform_load(i) for i in images]
