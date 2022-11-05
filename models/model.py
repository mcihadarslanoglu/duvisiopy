
import torch
import time
import sys
import time
import sklearn.metrics


class Model():

    def __init__(self,
                 model
                 ):
        self.model = model

    """
    def __new__(self, cls):
        for attr in dir(self):
            if attr not in dir(cls):
                setattr(cls, attr, getattr(self, attr))
        return cls
    """

    def compile(self,
                loss_function,
                optimizer):
        self.loss_function = loss_function
        self.optimizer = optimizer

    def fit(self,
            train_loader,
            validation_loader=None,
            epochs=1,
            verbose = 1
            ):

        train_loss_history = []
        validation_loss_history = []

        train_accuracy_history = []
        validation_accuracy_history = []
        self.model.train()
        for index, epoch in enumerate(range(1, epochs + 1)):

            epoch_train_loss = 0
            epoch_train_accuracy = 0

            print("\nEpoch {}/{}".format(epoch, epochs))

            for index, (inputs, labels) in enumerate(self.progressbar(train_loader,"")):
                #print(inputs)
                
                self.optimizer.zero_grad()
                output = self.model(inputs)
                
                loss = self.loss_function(output, labels)
                
                #print(output)
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
                _, preds = torch.max(output, 1)
                #return output

                #print(preds,labels)
                #time.sleep(0.5)
                
                epoch_train_accuracy += torch.sum(preds ==
                 
                                                  labels.data).item()
            
            
            epoch_train_loss = epoch_train_loss/len(train_loader.dataset)
            epoch_train_accuracy = epoch_train_accuracy/len(train_loader.dataset)
            train_loss_history.append(epoch_train_loss)
            train_accuracy_history.append(epoch_train_accuracy)

            if validation_loader:
                with torch.no_grad():
                    epoch_validation_loss = 0
                    epoch_validation_accuracy = 0
                    for inputs, label in validation_loader:

                        output = self.model(inputs)

                        epoch_validation_loss = self.loss_function(
                            output, label)

                        _, preds = torch.max(output, 1)

                        epoch_validation_loss += epoch_validation_loss.item()
                        epoch_validation_accuracy += torch.sum(
                            preds == label.data)/len(validation_loader.dataset)

                    validation_loss_history.append(epoch_train_loss)
                    validation_accuracy_history.append(epoch_train_accuracy)

                after_progressbar = " acc: {:.4f} - loss: {:.4e} - val_acc: {:.4f} - val_loss: {:.4e}".format(epoch_train_accuracy,
                                                                                                              epoch_train_loss,
                                                                                                              epoch_validation_accuracy,
                                                                                                              epoch_validation_loss)
                print(after_progressbar)
                continue

            after_progressbar = " acc: {:.4f} - loss: {:.4e}".format(epoch_train_accuracy,
                                                                     epoch_train_loss)
            print(after_progressbar)

            print(after_progressbar, file=sys.stdout, flush=True)

        return {"acc": train_accuracy_history, "loss": train_loss_history, "val_acc": validation_accuracy_history, "val_loss": validation_loss_history}

    def evaluate(self, test_loader, is_dump=True, is_return=True):

        pred_labels = []
        labels= []
        for index, (inputs, label) in enumerate(self.progressbar(test_loader, "")):
            with torch.no_grad():
                output = self.model(inputs)
                loss = self.loss_function(output, label)
                

                _, preds = torch.max(output, 1)
                pred_labels.extend(preds.tolist())
                labels.extend(label.tolist())
                
        
        classification_report = sklearn.metrics.classification_report(
            labels, pred_labels)
        confusion_matrix = sklearn.metrics.confusion_matrix(
            labels, pred_labels)
        if is_dump:
            print("           --------- classification report ----------")
            print(classification_report)

            print("--- confusion matrix ---")
            print(confusion_matrix)
        if is_return:
            return {"classification_reprt": classification_report,
                    "confusion_matrix": confusion_matrix}

    def progressbar(self, it,  prefix="", postfix="=>", after_progressbar="", size=60, out=sys.stdout):  # Python3.3+
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
