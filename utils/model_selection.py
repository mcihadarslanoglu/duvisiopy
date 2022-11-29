import duvisiopy.utils.data
import math
import sklearn.model_selection
import torch.utils.data
import torch
import torch.utils.data
import numpy


def cross_validate(model, dataset, n_fold, **model_parameters):
    Kfold = Kfold(n_fold, shuffle=True)
    datasets = Kfold.split(dataset)

    for index, test_dataset in enumerate(datasets):
        train_dataset = duvisiopy.utils.data.Dataset()

        tmp_dataset = datasets.copy()
        tmp_dataset.pop(index)
        train_dataset = train_dataset.concat(tmp_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

        model.fit()
        model.evaluate()


class Kfold():
    def __init__(self, fold_number, shuffle=True) -> None:

        self.fold_number = fold_number
        self.shuffe = shuffle

    def split(self, dataset):

        folds = []
        data, label = dataset.__getlist__()
        sample_number = len(dataset)
        step_number = math.ceil(sample_number/self.fold_number)
        start_point = 0
        for step in numpy.full(self.fold_number, step_number):
            X = data[start_point:start_point + step]
            y = label[start_point:start_point + step]

            start_point = start_point + step

            tmp_dataset = duvisiopy.utils.data.Dataset()
            tmp_dataset.__setlabel__(y)
            tmp_dataset.__setdata__(X)
            folds.append(tmp_dataset)

        return folds
