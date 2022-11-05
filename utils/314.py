import cv2
import os
import torch.utils.data


class Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 path,
                 sample_counts=None,
                 transforms=None,
                 verbobse=0
                 ):
        """
        Wraps the dataset

        Parameters
        ----------
        path : str
            absolute or relative path of the dataset
        transforms : torchvision.transforms, optional
            , by default None
        """
        super(Dataset).__init__()
        self.data = []
        self.transforms = transforms

        for label, folder in enumerate(os.listdir(path)):

            if sample_counts:
                sample_count = sample_counts.get(label, -1)

        for file_name in os.listdir(os.path.join(path, folder))[0:sample_count]:
            self.data.append(
                [os.path.join(path, folder, file_name), label])

    def __len__(self):
        """
        Returns length of the dataset.

        Returns
        -------
        int
            Length of dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a dotum of the dataset.

        Parameters
        ----------
        idx : int
            index of dotum.

        Returns
        -------
        tuple-*
            (image,label)
        """

        file = self.data[idx]
        image_path = file[0]
        label = file[1]
        image = cv2.imread(image_path)
        if self.transforms:
            image = self.transforms(image)
        return image, label
