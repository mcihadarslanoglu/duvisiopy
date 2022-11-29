import time
import os
import json
import torch


class Callbacks:

    def __init__(self) -> None:
        pass

    def on_batch_begin(self):
        raise NotImplementedError

    def on_batch_end(self):
        raise NotImplementedError

    def on_epoch_begin(self):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError


class AutoSave(Callbacks):

    def __init__(self, at_epoch, save_path='results') -> None:
        super().__init__()

        self.at_epoch = at_epoch
        self.timestamp = str(int(time.time()))
        self.save_path = os.path.join(save_path, self.timestamp)

    def on_epoch_end(self, logs):
        print(logs)
        print(logs['epoch'] % self.at_epoch)
        if logs['epoch'] % self.at_epoch != 0:
            return False

        model = logs['model']
        metadata = {}

        metadata.update({'model': logs['model_metadata']})

        metadata['train'] = logs['train_loader_metadata']
        metadata['train'].update({'epochs': logs['epoch'],
                                  "acc": logs['train_acc'], "loss": logs['train_loss']})

        if 'validation_loader_metadata' in list(metadata.keys()):
            metadata['validation'] = logs['validation_loader_metadata']
            metadata['validation'].update(
                {"acc": logs['val_acc'], "loss": logs['val_loss']})

        os.makedirs(os.path.join(self.save_path, "train"), exist_ok=True)

        os.makedirs(os.path.join(self.save_path, "train",
                                 ), exist_ok=True)

        json.dump(metadata, open(os.path.join(
            self.save_path, "train",  'history.json'), 'w'), indent=4)
        torch.save(model.state_dict(), os.path.join(
            self.save_path, 'train', 'model.pth'))
        return True
