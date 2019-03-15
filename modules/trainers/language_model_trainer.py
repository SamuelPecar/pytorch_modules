import math
import time
import torch
import torch.nn.functional as F

from tqdm import tqdm

from .trainer import Trainer
from ..common.utils import vectorize, repackage_hidden


def batch_train_print(epoch, batch, num_batches, total_loss, log_interval, start_time):
    current_loss = total_loss / log_interval
    elapsed = time.time() - start_time

    try:
        print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.5f} | ppl {:5.5f}'.format(epoch, batch, num_batches, elapsed * 1000 / log_interval, current_loss,
                                                                                                           math.pow(2, current_loss)))
    except OverflowError:
        print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.5f}'.format(epoch, batch, num_batches, elapsed * 1000 / log_interval, current_loss))


class LanguageModelTrainer(Trainer):
    def __init__(self, model, criterion, optimizer, device, vocab, batch_size, log_interval=200, **kwargs):
        super(LanguageModelTrainer, self).__init__()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.vocab = vocab

        self.batch_size = batch_size

        self.log_interval = log_interval

        self.hidden = None

    def train_model(self, epoch, data):
        epoch_loss = 0
        self.model.train()

        num_batches = math.ceil(len(data.dataset) / self.batch_size)
        start_time = time.time()
        self.hidden = self.model.encoder.init_hidden(self.batch_size)

        for i_batch, batch in enumerate(tqdm(data), 1):
            self.optimizer.zero_grad()
            self.hidden = repackage_hidden(self.hidden)

            outputs, targets, loss = self.__process_batch(batch)

            epoch_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            loss.backward()
            self.optimizer.step()

            if i_batch % self.log_interval == 0 and i_batch > 0:
                batch_train_print(epoch, i_batch, num_batches, epoch_loss, self.log_interval, start_time)
                epoch_loss = 0
                start_time = time.time()

    def evaluate_model(self, data):
        epoch_loss = 0
        self.model.eval()
        self.hidden = self.model.encoder.init_hidden(batch_size)

        with torch.no_grad():
            for i_batch, batch in enumerate(tqdm(data_source), 1):
                self.hidden = repackage_hidden(self.hidden)

                outputs, targets, loss = self.__process_batch(batch)

                epoch_loss += loss.item()

        return total_loss / i_batch

    def __process_batch(self, batch):

        inputs, targets, lengths = batch

        inputs, mask = vectorize(inputs, self.vocab.word2idx, self.device)
        targets, _ = vectorize(targets, self.vocab.word2idx, self.device)

        outputs, self.hidden = self.model(inputs=inputs, mask=mask, hidden=self.hidden, lengths=lengths.to(self.device))

        if type(self.criterion) == torch.nn.modules.loss.NLLLoss:
            outputs = F.log_softmax(outputs, dim=-1)
            loss = self.criterion(outputs.view(-1, outputs.size()[2]), targets.view(-1))
        else:
            loss = self.criterion(outputs.view(-1, outputs.size()[2]), targets.view(-1))

        return outputs, targets, loss


