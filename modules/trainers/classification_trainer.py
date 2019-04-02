import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from .trainer import Trainer


class ClassificationTrainer(Trainer):
    def __init__(self, model, criterion, optimizer, device, **kwargs):
        super(ClassificationTrainer, self).__init__()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.hidden = None

    def train_model(self, data):
        epoch_loss = 0
        self.model.train()

        for i_batch, batch in enumerate(tqdm(data), 1):
            self.optimizer.zero_grad()

            outputs, targets, loss = self.__process_batch(batch)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / i_batch

    def evaluate_model(self, data):
        epoch_loss = 0

        self.model.eval()

        predictions = []
        gold_labels = []

        with torch.no_grad():
            for i_batch, batch in enumerate(tqdm(data), 1):
                outputs, targets, loss = self.__process_batch(batch)

                outputs = F.softmax(outputs, dim=1)

                predictions.append(outputs)
                gold_labels.append(targets)

                epoch_loss += loss.item()

            predictions = np.array(torch.cat(predictions, dim=0).cpu())
            predicted = np.argmax(predictions, 1)
            labels = np.array(torch.cat(gold_labels, dim=0).cpu())

        return epoch_loss / i_batch, predicted, predictions, labels

    def __process_batch(self, batch):

        inputs, targets, lengths = batch

        outputs, self.hidden = self.model(inputs=inputs, mask=None, hidden=None, lengths=lengths.to(self.device))

        if type(self.criterion) == torch.nn.modules.loss.NLLLoss:
            loss = self.criterion(F.log_softmax(outputs, dim=1), targets.to(self.device))
        else:
            loss = self.criterion(outputs, targets.to(self.device))

        return outputs, targets, loss
