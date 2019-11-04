import torch.nn as nn
from transformers import *
from torch.nn.utils.rnn import pad_sequence

try:
    from config import device
except:
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransformerEmbeddings(nn.Module):
    """
    Proxy for transformer embeddings.
    https://github.com/huggingface/transformers
    Args:
        model_name str: name of required model (e.g. BERT)
        weights_shortcut str: shortcut for pretrained weights e.g. bert-base-uncased
        embedding_dropout float: value of dropout
    """

    def __init__(self, model_name, weights_shortcut, embedding_dropout=0., **kwargs):
        super(TransformerEmbeddings, self).__init__()

        self.dropout = nn.Dropout(p=embedding_dropout)
        self.transformer, self.tokenizer = self.get_model(model_name, weights_shortcut)
        # TODO set for bert large
        self.embedding_dim = 1024

    def get_model(self, model_name, pretrained_weights):
        MODELS = {"bert": (BertModel, BertTokenizer),
                  "gpt": (OpenAIGPTModel, OpenAIGPTTokenizer),
                  "gpt2": (GPT2Model, GPT2Tokenizer),
                  "ctrl": (CTRLModel, CTRLTokenizer),
                  "transforxl": (TransfoXLModel, TransfoXLTokenizer),
                  "xlnet": (XLNetModel, XLNetTokenizer),
                  "xlm": (XLMModel, XLMTokenizer),
                  "distilbert": (DistilBertModel, DistilBertTokenizer),
                  "Roberta": (RobertaModel, RobertaTokenizer)}

        MODEL, TOKENIZER = MODELS[model_name]

        # model = MODEL.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True).to(device)
        model = MODEL.from_pretrained(pretrained_weights).to(device)
        tokenizer = TOKENIZER.from_pretrained(pretrained_weights)

        return model, tokenizer

    def tokenize(self, sentences, joined_tokens=False):
        sentences_ = []

        for sentence in sentences:
            if not joined_tokens:
                sentence = " ".join(sentence)
            tokenized_sentence = self.tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt')[0]
            sentences_.append(tokenized_sentence)

        return pad_sequence(sentences_, batch_first=True)

    def forward(self, sentences, joined_tokens=False):
        if not joined_tokens:
            sentences = self.join_tokens(sentences)
        sentences = self.tokenize(sentences, joined_tokens)

        last_hidden_states = self.transformer(sentences)[0]

        return last_hidden_states, None
