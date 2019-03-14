import re
from .regex_expressions import *

from allennlp.data.tokenizers.word_tokenizer import WordTokenizer

tokenizer = WordTokenizer(end_tokens=['<EOS>'])


# TODO - extend settings and add emoji end emoticon processing
class Preprocessing(object):
    """
    Module for text pre-processing
    """

    def __init__(self, **kwargs):
        self.char_clean = kwargs.get('char_cleaning', False)
        self.char_normalize = kwargs.get('char_normalize', False)
        self.word_normalize = kwargs.get('word_normalization', False)
        self.expand = kwargs.get('expand', False)
        self.escape_punctuation = kwargs.get('escape_punctuation', False)
        self.negation = kwargs.get('negation', False)

    def split_text(self, text):
        return text.split()

    def tokenize(self, text):
        tokens = tokenizer.tokenize(text)
        return [t.text for t in tokens]

    def process_text(self, text):

        tokens = tokenizer.tokenize(text)
        text = ' '.join([t.text for t in tokens])

        if self.char_clean:
            text = self.char_cleaning(text)
        if self.char_normalize:
            text = self.char_normalization(text)
        if self.word_normalize:
            text = self.word_normalization(text)
        if self.expand:
            text = self.phrase_expanding(text)
        if self.escape_punctuation:
            text = self.escaping(text)
        if self.negation:
            text = self.word_negation(text)

        return text.split()

    @staticmethod
    def char_cleaning(text):
        # ranges from http://jrgraphix.net/research/unicode.php
        text = re.sub('[\u0370-\u03ff]', '', text)  # Greek and Coptic
        text = re.sub('[\u0400-\u052f]', '', text)  # Cyrillic and Cyrillic Supplementary
        text = re.sub('[\u2500-\u257f]', '', text)  # Box Drawing
        text = re.sub('[\u2e80-\u4dff]', '', text)  # from CJK Radicals Supplement
        text = re.sub('[\u4e00-\u9fff]', '', text)  # CJK Unified Ideographs
        text = re.sub('[\ue000-\uf8ff]', '', text)  # Private Use Area
        text = re.sub('[\uff00-\uffef]', '', text)  # Halfwidth and Fullwidth Forms
        text = re.sub('[\ufe30-\ufe4f]', '', text)  # CJK Compatibility Forms

        return text

    @staticmethod
    def char_normalization(text):
        text = re.sub(INVISIBLE_REGEX, '', text)
        text = re.sub(QUOTATION_REGEX, '\"', text)
        text = re.sub(APOSTROPHE_REGEX, '\'', text)
        text = re.sub(r"\s+", " ", text)

        return text

    @staticmethod
    # TODO
    def word_normalization(text):
        # text = re.sub(EMAIL_REGEX, ' <EMAIL> ', text)
        # text = re.sub(URL_REGEX, ' <URL> ', text)
        # text = re.sub(USER_REGEX, ' @USER ', text)

        text = re.sub(PRICE_REGEX, r" <PRICE> ", text)
        text = re.sub(TIME_REGEX, r" <TIME> ", text)
        text = re.sub(DATE_REGEX, r" <DATE> ", text)

        text = re.sub(r"([a-zA-Z]+)([0-9]+)", r"\1 \2", text)
        text = re.sub(r"([0-9]+)([a-zA-Z]+)", r"\1 \2", text)
        text = re.sub(r" [0-9]+ ", r" <NUMBER> ", text)

        return text

    @staticmethod
    def phrase_expanding(text):
        text = re.sub(r"(\b)([Ii]) 'm", r"\1\2 am", text)
        text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou) 're", r"\1\2 are", text)
        text = re.sub(r"(\b)([Ll]et) 's", r"\1\2 us", text)
        text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou) 'll", r"\1\2 will", text)
        text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou) 've", r"\1\2 have", text)

        return text

    @staticmethod
    def word_negation(text):
        text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]as|[Ww]ould) n't", r"\1\2 not", text)
        text = re.sub(r"(\b)([Cc]a) n't", r"\1\2n not", text)
        text = re.sub(r"(\b)([Ww]) on't", r"\1\2ill not", text)
        text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
        text = re.sub(r" n't ", r" not ", text)

        return text

    @staticmethod
    def escaping(text):
        text = re.sub(r"([‼.,;:?!…])+", r" \1 ", text)
        text = re.sub(r"([()])+", r" \1 ", text)
        text = re.sub(r"[-]+", r" - ", text)
        text = re.sub(r"[_]+", r" _ ", text)
        text = re.sub(r"[=]+", r" = ", text)
        text = re.sub(r"[\&]+", r" \& ", text)
        text = re.sub(r"[\+]+", r" \+ ", text)

        return text
