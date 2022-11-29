
import torch
import time
import sys
import time
import sklearn.metrics
import time
import os
import seaborn
import pandas
import matplotlib.pyplot
import tqdm
import time
import json


class Model():

    def __init__(self,
                 model,
                 base_model_parameters=None
                 ):
        self.model = model
        self.base_model_parameters = base_model_parameters
        self.timestamp = str(int(time.time()))
        self.save_path = os.path.join('results', self.timestamp)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def compile(self,
                loss_function,
                optimizer):
        """compile the model

        Parameters
        ----------
        loss_function : torch.nn.{any_loss_function}
            loss function of the model
        optimizer : torch.optim.{any_optimizer_function}
            optimizer function of the model
        """
        self.loss_function = loss_function
        self.optimizer = optimizer

    def fit(self,
            train_loader,
            validation_loader=None,
            epochs=2,
            verbose=1,
            save=True,
            callbacks=None
            ):
        """
        Train the model by provided parameters.

        Parameters
        ----------
        train_loader : torch.utils.data
            the loader that will be used training process
        validation_loader : torch.utils.data, optional
            the loader twhat will be used to validate model, by default None
        epochs : int, optional
            Number of epoch, by default 1
        verbose : int, optional
            1: prints progress bar, loss and accuracy, by default 1

        Returns
        -------
        dict
            return {"acc": train_accuracy_history, "loss": train_loss_history, "val_acc": validation_accuracy_history, "val_loss": validation_loss_history}
        """

        train_loss_history = []
        validation_loss_history = []

        train_accuracy_history = []
        validation_accuracy_history = []

        train_loader_len = len(train_loader.dataset)

        for epoch in range(1, epochs + 1):
            epoch_train_loss = 0
            epoch_train_accuracy = 0
            loop = tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader), unit='batch', colour='cyan')
            for index, (inputs, labels) in loop:

                self.optimizer.zero_grad()
                output = self.model(inputs)

                loss = self.loss_function(output, labels)

                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item()/train_loader_len
                _, preds = torch.max(output, 1)

                epoch_train_accuracy += torch.sum(preds ==

                                                  labels.data).item()/train_loader_len
                loop.set_description(
                    "Epoch {}/{}".format(epoch, epochs))
                loop.set_postfix(
                    loss=loss.item(), acc=epoch_train_accuracy)

            train_loss_history.append(epoch_train_loss)
            train_accuracy_history.append(epoch_train_accuracy)

            logs = {
                'train_acc': train_accuracy_history,
                'train_loss': train_loss_history,
                'train_loader_metadata': train_loader.dataset.metadata,
                'epochs': epochs,
                'epoch': epoch,
                'model': self.model,
                'model_metadata': self.get_metadata()
            }

            if validation_loader:
                logs.update({
                    'val_acc': validation_accuracy_history,
                    'val_loss': validation_loss_history,
                    'validation_loader_metadata': validation_loader.dataset.metadata,
                })

                validation_loader_len = len(validation_loader.dataset)

                with torch.no_grad():
                    epoch_validation_loss = 0
                    epoch_validation_accuracy = 0
                    for inputs, label in validation_loader:

                        output = self.model(inputs)

                        epoch_validation_loss = self.loss_function(
                            output, label)

                        _, preds = torch.max(output, 1)

                        epoch_validation_loss += epoch_validation_loss.item()/validation_loader_len
                        epoch_validation_accuracy += torch.sum(
                            preds == label.data)/validation_loader_len

                validation_loss_history.append(epoch_validation_loss.item())
                validation_accuracy_history.append(
                    epoch_validation_accuracy.item())

                print('acc: {} loss: {} val_acc: {} val_loss: {}'.format(
                    epoch_train_accuracy, epoch_train_loss, epoch_validation_accuracy, epoch_validation_loss))
            if callbacks:

                for callback in callbacks:
                    callback.on_epoch_end(logs)

        return {'epochs': epochs,
                "acc": train_accuracy_history, "loss": train_loss_history,
                "val_acc": validation_accuracy_history, "val_loss": validation_loss_history}

        """
            if save:
                metadata = {}

                model_metadata = self.get_metadata()
                metadata.update({'model': model_metadata})

                metadata['train'] = train_loader.dataset.metadata
                metadata['train'].update({'epochs': epochs,
                                        "acc": train_accuracy_history, "loss": train_loss_history})

                if validation_loader:
                    metadata['validation'] = validation_loader.dataset.metadata
                    metadata['validation'].update(
                        {"acc": validation_accuracy_history, "loss": validation_loss_history})

                os.makedirs(os.path.join(self.save_path, "train"), exist_ok=True)
                tmp_timestamp = str(int(time.time()))
                os.makedirs(os.path.join(self.save_path, "train",
                            tmp_timestamp), exist_ok=True)

                json.dump(metadata, open(os.path.join(
                    self.save_path, "train", tmp_timestamp, 'history.json'), 'w'), indent=4)
                torch.save(self.model.state_dict(), os.path.join(
                    self.save_path, 'train', tmp_timestamp, 'model.pth'))

            """

    def evaluate(self, test_loader, is_dump=True, is_return=True, is_save=True, save_path=None):
        """ Evalueates the model and saves its results

        Parameters
        ----------
        test_loader : torch.Dataset
            torch dataloader which is desired to evaluate.
        is_dump : bool, optional
            if True, dump the results into screen, by default True
        is_return : bool, optional
            if True, returns the results., by default True
        is_save : bool, optional
            if True, saves results to desired path. Results are saved into results/timestamp in case of path is not specificied., by default True
        save_path : str, optional
            relative path to save results, by default None

        Returns
        -------
        dict
            Returns {'classification_reprt':classification_report,'confusion_matrix':confusion_matrix} if is_return is True.
        """

        ## CREATING AND DUMPİNG ##
        metadata = {}
        metadata['test_loader'] = test_loader.dataset.metadata
        pred_labels = []
        labels = []
        test_loader_len = len(test_loader)
        test_loader = tqdm.tqdm(test_loader)
        acc_total = 0
        loss_total = 0

        for index, (inputs, label) in enumerate(test_loader):
            with torch.no_grad():
                output = self.model(inputs.to(self.device))
                loss = self.loss_function(output, label)

                _, preds = torch.max(output, 1)
                pred_labels.extend(preds.tolist())
                labels.extend(label.tolist())

                acc_total = acc_total + \
                    (torch.sum(preds == label)/test_loader_len).item()
                loss_total = loss_total + loss.item()

                test_loader.set_postfix(
                    loss=loss_total, acc=acc_total)

        classification_report = sklearn.metrics.classification_report(
            labels, pred_labels, output_dict=True)
        confusion_matrix = sklearn.metrics.confusion_matrix(
            labels, pred_labels)
        if is_dump:
            print("           --------- classification report ----------")
            print(classification_report)

            print("--- confusion matrix ---")
            print(confusion_matrix)
        #########################################################################################

        ## SAVE THE RESULTS ##
        if is_save:
            tmp_timestamp = str(int(time.time()))
            if not save_path:
                save_path = os.path.join(
                    self.save_path, "evaluate", tmp_timestamp)
            tmp_path = ""
            for path in save_path.split("/"):
                tmp_path = os.path.join(tmp_path, path)

                if not os.path.isdir(tmp_path):
                    os.makedirs(tmp_path, exist_ok=True)

            os.makedirs(os.path.join(save_path, "plot"), exist_ok=True)

            print("Sonuçlar {} klasörüne kayıt edildi.".format(save_path))
            seaborn.set(font_scale=1.4)
            seaborn.set(rc={'figure.figsize': (11.7, 8.27)})
            seaborn.heatmap(pandas.DataFrame(
                classification_report).iloc[:-1, :].T, annot=True).get_figure().savefig(os.path.join(save_path, "plot", "classification_report.png"))

            matplotlib.pyplot.clf()

            seaborn.heatmap(pandas.DataFrame(confusion_matrix), annot=True).get_figure().savefig(
                os.path.join(save_path, "plot", "confusion_matrix.png"))
            metadata.update({'acc': acc_total, 'loss': loss_total})
            json.dump(metadata, open(os.path.join(
                self.save_path, "evaluate", tmp_timestamp, 'history.json'), 'w'), indent=4)

        #########################################################################################

        ## RETURN THE RESULTS ##
        if is_return:
            return {"classification_reprt": classification_report,
                    "confusion_matrix": confusion_matrix}
        #########################################################################################

    def freeze_layer(self, layers):
        """
        Freeze any desired layers

        Parameters
        ----------
        layers : str,list
            Layer names which is desired to freeze.
        """

        for name, layer in self.model.named_parameters():

            if name.split(".")[0] in layers:
                layer.requires_grad = False

    def unfreeze_layer(self, layers):
        """
        Unfreeze any desired layers

        Parameters
        ----------
        layers : str,list
            Layer names which is desired to unfreeze.
        """

        for name, layer in self.model.named_parameters():

            if name.split(".")[0] in layers:
                layer.requires_grad = True

    def get_metadata(self):
        metadata = {}
        frozen_layers = [
            name for name, layer in self.model.named_parameters() if not layer.requires_grad]
        metadata['name'] = self.model.__class__.__name__
        if self.base_model_parameters:
            metadata.update(self.base_model_parameters)
        metadata['loss_function'] = {
            'name': self.loss_function.__class__.__name__}
        metadata['optimizer'] = {'name': self.optimizer.__class__.__name__}
        metadata['optimizer'].update(self.optimizer.defaults)
        metadata['frozen_layers'] = frozen_layers

        return metadata

    def progressbar(self, it,  prefix="", postfix="=>", after_progressbar="", size=60, out=sys.stdout):
        count = len(it)
        sample_size = len(it.dataset)
        batch_size = it.batch_size

        def show(j, batch_size, sample_size, _after_progressbar):
            x = int(size*j/count)

            print("{}{}/{} [{}{}{}] {}".format(prefix,
                                               j*batch_size,
                                               sample_size,
                                               "#"*x,
                                               postfix,
                                               "."*(size-x),
                                               _after_progressbar
                                               ),
                  end="\r",
                  file=out,
                  flush=True)

        show(0, batch_size, sample_size, "")
        total_time = 0
        for i, item in enumerate(it):
            start_time = time.time()
            yield item
            end_time = time.time()
            delta_time = end_time - start_time
            _after_progressbar = "- ETA: {:.0f} {}".format(
                delta_time*(count - i), after_progressbar)
            show(i+1, batch_size, sample_size,
                 _after_progressbar)
            total_time = total_time + delta_time

        print(" "*100, end="\r")
        print("{}/{} [{}] {:.3f}s".format(sample_size, sample_size,
              "#"*size, total_time), flush=True, file=out, end="")
