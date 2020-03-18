from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
import keras.utils as ku
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

from tensorflow.compat.v1 import set_random_seed
from numpy.random import seed
set_random_seed(2)
seed(1)

import re
import numpy as np
import logging
import sys

FORMAT = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)


class TrainingBundle(object):
    __DATA_TYPE__ = type(list())
    __META_TYPE__ = type(dict())

    def __init__(self, data, meta):
        """Type check"""
        if data and not isinstance(data, TrainingBundle.__DATA_TYPE__):
            raise TypeError("TrainingBundle.data expected {} got {}".format(TrainingBundle.__DATA_TYPE__, type(data)))
        if meta and not isinstance(meta, TrainingBundle.__META_TYPE__):
            raise TypeError("TrainingBundle.meta expected {} got {}".format(TrainingBundle.__META_TYPE__, type(meta)))

        if not data:
            raise TypeError("data is required")

        if not meta:
            raise TypeError("meta is required")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.__meta = meta
        self.__total_word_count = 0
        self.__max_seq_len = 0
        self.__tokenizer = Tokenizer()
        self.__data = self.__tokenize_data__(data)
        self.__predictors, self.__label = self.__generate_padded_sequences__(self.__data)

    def __generate_padded_sequences__(self, data):
        self.logger.debug("Padding ngram sequences")
        max_len = len(max(data, key=len))
        self.__max_seq_len = max_len
        data = np.array(pad_sequences(data, maxlen=max_len, padding='pre'))
        predictors, label = data[:, :-1], data[:, -1]
        label = ku.to_categorical(label, num_classes=self.__total_word_count)
        return predictors, label

    @property
    def tokenizer(self):
        return self.__tokenizer

    def __tokenize_data__(self, data) -> list:
        self.logger.debug("Tokenizing data")
        tokenizer = self.tokenizer
        tokenizer.fit_on_texts(texts=data)
        self.__total_word_count = len(tokenizer.word_index) + 1

        _d = []
        for text_collection in data:
            text_collection = re.sub(r'\n|\t|\[.*?\]', ' ', text_collection)
            if text_collection:
                toks = tokenizer.texts_to_sequences([text_collection])[0]
                _d.extend([toks[:i+1] for i in range(1, len(toks))])   # Add ngrams to data
        return _d

    @property
    def word_count(self):
        return self.__total_word_count

    @property
    def max_sequence_length(self):
        return self.__max_seq_len

    @property
    def predictors(self):
        return self.__predictors

    @property
    def label(self):
        return self.__label


class PredictiveModel(object):

    def __init__(self, training_bundle: TrainingBundle):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.__model = Sequential()
        self.__model.add(Embedding(training_bundle.word_count, 3, input_length=training_bundle.max_sequence_length - 1))
        self.__model.add(LSTM(100))
        self.__model.add(Dropout(0.1))
        self.__model.add(Dense(training_bundle.word_count, activation='softmax'))
        opt = Adam()
        self.logger.debug("Compiling model..")
        self.__model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.logger.debug("Training model..")
        self.__model.fit(training_bundle.predictors, training_bundle.label, epochs=100, verbose=5)

    def write_model_to_file(self, fp):
        self.logger.debug("Writing model to file: '{}'".format(fp))
        model_json = self.__model.to_json()
        with open(fp, 'w') as json_file:
            json_file.write(model_json)
