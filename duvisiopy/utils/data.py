
import torch.utils.data
import os
import cv2
import numpy
import sklearn.utils


class Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 path=None,
                 sample_counts=None,
                 transforms=None,
                 verbose=0,
                 shuffle=True
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
        self.path = path
        self.verbose = verbose

        self.transforms = transforms
        self.metadata = self.get_metadata()

        sample_count = -1

        if path:
            if verbose == 1:
                print("Veriler okunuyor...\n")
                folders = os.listdir(path)
                folders.sort()

            for label, folder in enumerate(folders):
                successful_data = 1

                if sample_counts:
                    sample_count = sample_counts.get(folder, -1)

                for file_name in os.listdir(os.path.join(path, folder))[0:sample_count]:

                    """
                    self.data.append(
                        [os.path.join(path, folder, file_name), label])

                    """
                    image = cv2.imread(os.path.join(
                        path, folder, file_name)).astype(numpy.float32)
                    if transforms:
                        image = transforms(image)

                    self.data.append(image)
                    self.labels.append(label)
                    successful_data = successful_data + 1

                if verbose == 1:
                    print("{} sınıfına ait {} veri okundu. Sınıfın etiketi {}".format(
                        folder, successful_data, label))
                self.metadata['classes'].append(
                    {'name': folder, 'number': successful_data, 'label': label})
            if verbose == 1:
                print("\nToplamam okunan veri sayısı {}".format(len(self.data)))
            if shuffle:
                self.data, self.labels = sklearn.utils.shuffle(
                    self.data, self.labels)

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

        return image, label  # torch.tensor(label).type(torch.LongTensor)

    def __getlist__(self):
        return self.data, self.labels

    def __setdata__(self, data):
        self.data = data

    def __setlabel__(self, labels):
        self.labels = labels

    def get_metadata(self):

        metadata = {'dataset_path': self.path,
                    'classes': [], 'verbose': self.verbose}
        if self.transforms:
            transforms_list = []
            for transform in self.transforms.transforms:
                parameters = {'name': transform.__class__.__name__}
                for parameter in transform.__dict__:
                    if not parameter.startswith("_"):
                        parameters.update(
                            {parameter: transform.__dict__[parameter].__str__()})

                transforms_list.append(parameters)

            metadata['transforms'] = transforms_list

        return metadata

    def concat(self, datasets):
        for dataset in datasets:
            self.data.extend(dataset.data)
            self.labels.extend(dataset.labels)
