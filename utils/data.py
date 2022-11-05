
import torch.utils.data
import os
import cv2
import numpy

class Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 path,
                 sample_counts=None,
                 transforms=None,
                 verbose=0
                 ):
        """
        Wraps the dataset

        Parameters
        ----------
        path : str
            absolute or relative path of the dataset
        transforms : torchvision.transforms, optional
            , by default None
        sample_counts : dict
            sample counts can be declared like {"folder_name1":count1, "folder_name2":count2}. Not declared classes count is set all by default.
        verbose : int
            1: prints count of each succesful read data.
        """
        super(Dataset).__init__()
        self.data = []
        self.labels = []
        self.transforms = transforms
        sample_count = -1
        if verbose == 1:
            print("Veriler okunuyor...\n")

        for label, folder in enumerate(os.listdir(path)):
            sucessful_data = 1

            if sample_counts:
                sample_count = sample_counts.get(folder, -1)

            for file_name in os.listdir(os.path.join(path, folder))[0:sample_count]:
                
                """
                self.data.append(
                    [os.path.join(path, folder, file_name), label])
                
                """
                self.data.append(cv2.imread(os.path.join(path,folder,file_name)).astype(numpy.float32))
                self.labels.append(label)
                sucessful_data = sucessful_data + 1

            if verbose == 1:
                print("{} sınıfına ait {} veri okundu. Sınıfın etiketi {}".format(
                    folder, sucessful_data, label))
        if verbose == 1:
            print("\nToplamam okunan veri sayısı {}".format(len(self.data)))
            

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
        tuple
            (image,label)
        """

        image = self.data[idx]
        label = self.labels[idx]
  
  
        if self.transforms:
            image = self.transforms(image)
            
        return image, label#torch.tensor(label).type(torch.LongTensor)

