from abc import ABC, abstractmethod
from math import log10

from VocabularyFactory import CaseInsensitiveAlphabetChars


class TrainingModelFactory:
    @staticmethod
    def get_nb_training_model(vocabulary, ngram_size, smoothing_value, training_file):
        if ngram_size == '1':
            return UnigramTrainingModel(vocabulary, smoothing_value, training_file)

        if ngram_size == '2':
            return BigramTrainingModel(vocabulary, smoothing_value, training_file)

        if ngram_size == '3':
            return TrigramTrainingModel(vocabulary, smoothing_value, training_file)


class TrainingModel(ABC):
    def __init__(self, vocabulary, smoothing_value, training_file):
        self.vocabulary = vocabulary
        self.vocabulary_size = vocabulary.get_size()
        self.smoothing_value = smoothing_value
        self.training_file = training_file
        self.ngram_frequencies = dict()
        self.language_data = {
            'eu': {
                'doc_freq': 0,
            },
            'ca': {
                'doc_freq': 0,
            },
            'gl': {
                'doc_freq': 0,
            },
            'es': {
                'doc_freq': 0,
            },
            'en': {
                'doc_freq': 0,
            },
            'pt': {
                'doc_freq': 0,
            }
        }

        for language in self.language_data.keys():
            self.ngram_frequencies[language] = dict()
            self.ngram_frequencies[language]['total_count'] = 0

    def train(self):
        input_file = open(self.training_file, 'r', encoding="utf-8")
        for line in input_file:
            partitioned_line = line.split(maxsplit=3)
            language = partitioned_line[2]
            tweet = partitioned_line[3]

            self.language_data[language]['doc_freq'] += 1
            if isinstance(self.vocabulary, CaseInsensitiveAlphabetChars):
                tweet = tweet.lower()

            self.process_tweet(language, tweet)

    @abstractmethod
    def process_tweet(self, language, tweet):
        pass

    @abstractmethod
    def parse_tweet(self, tweet):
        pass

    @abstractmethod
    def get_ngram_probability(self, ngram, language):
        pass

    def get_probability_of_language(self, language):
        frequency_of_language = self.language_data[language]['doc_freq']
        total_doc_count = self.get_num_docs()
        return frequency_of_language / total_doc_count

    def get_num_docs(self):
        num_docs = 0
        for language_data in self.language_data.values():
            num_docs += language_data['doc_freq']
        return num_docs


class UnigramTrainingModel(TrainingModel):
    def __init__(self, vocabulary, smoothing_value, training_file):
        super().__init__(vocabulary, smoothing_value, training_file)

    def parse_tweet(self, tweet):
        unigrams = []
        for char in tweet:
            if self.vocabulary.is_in_vocabulary(char):
                unigrams.append(char)
        return unigrams

    def process_tweet(self, language, tweet):
        for unigram in self.parse_tweet(tweet):
            self.ngram_frequencies[language]['total_count'] += 1

            codepoint = ord(unigram)
            if codepoint not in self.ngram_frequencies[language]:
                self.ngram_frequencies[language][ord(unigram)] = 1
            else:
                self.ngram_frequencies[language][ord(unigram)] += 1

    def get_ngram_probability(self, ngram, language):
        freq_ngram_for_language = self.smoothing_value
        try:
            freq_ngram_for_language += self.ngram_frequencies[language][ord(ngram)]
        except KeyError:
            pass

        total_ngram_count_for_language = self.ngram_frequencies[language]['total_count'] + \
                                         self.smoothing_value * self.vocabulary_size

        probability = freq_ngram_for_language / total_ngram_count_for_language
        if probability != 0:
            return log10(probability)
        else:
            return float('-inf')  # in case there is no smoothing value, assign negative infinity


class BigramTrainingModel(TrainingModel):
    def __init__(self, vocabulary, smoothing_value, training_file):
        super().__init__(vocabulary, smoothing_value, training_file)

    def parse_tweet(self, tweet):
        bigrams = []
        if len(tweet) < 2:
            return bigrams

        for i in range(len(tweet) - 1):
            bigram = tweet[i] + tweet[i + 1]
            if all(self.vocabulary.is_in_vocabulary(char) for char in bigram):
                bigrams.append(bigram)
        return bigrams

    def process_tweet(self, language, tweet):
        for bigram in self.parse_tweet(tweet):
            self.ngram_frequencies[language]['total_count'] += 1

            codepoint1 = ord(bigram[0])
            codepoint2 = ord(bigram[1])

            if codepoint1 not in self.ngram_frequencies[language]:
                self.ngram_frequencies[language][codepoint1] = dict()
                self.ngram_frequencies[language][codepoint1][codepoint2] = 1
            else:
                if codepoint2 not in self.ngram_frequencies[language][codepoint1]:
                    self.ngram_frequencies[language][codepoint1][codepoint2] = 1
                else:
                    self.ngram_frequencies[language][codepoint1][codepoint2] += 1

    def get_ngram_probability(self, ngram, language):
        codepoint1 = ord(ngram[0])
        codepoint2 = ord(ngram[1])

        freq_ngram_for_language = self.smoothing_value
        try:
            freq_ngram_for_language += self.ngram_frequencies[language][codepoint1][codepoint2]
        except KeyError:
            pass

        total_ngram_count_for_language = self.ngram_frequencies[language]['total_count'] + \
                                         self.smoothing_value * (self.vocabulary_size ** 2)
        probability = freq_ngram_for_language / total_ngram_count_for_language

        if probability != 0:
            return log10(probability)
        else:
            return float('-inf')  # in case there is no smoothing value, assign negative infinity


class TrigramTrainingModel(TrainingModel):
    def __init__(self, vocabulary, smoothing_value, training_file):
        super().__init__(vocabulary, smoothing_value, training_file)

    def parse_tweet(self, tweet):
        trigrams = []
        if len(tweet) < 3:
            return trigrams

        for i in range(len(tweet) - 2):
            trigram = tweet[i] + tweet[i + 1] + tweet[i + 2]
            if all(self.vocabulary.is_in_vocabulary(char) for char in trigram):
                trigrams.append(trigram)
        return trigrams

    def process_tweet(self, language, tweet):
        for trigram in self.parse_tweet(tweet):
            self.ngram_frequencies[language]['total_count'] += 1

            codepoint1 = ord(trigram[0])
            codepoint2 = ord(trigram[1])
            codepoint3 = ord(trigram[2])

            if codepoint1 not in self.ngram_frequencies[language]:
                self.ngram_frequencies[language][codepoint1] = dict()
                self.ngram_frequencies[language][codepoint1][codepoint2] = dict()
                self.ngram_frequencies[language][codepoint1][codepoint2][codepoint3] = 1
            else:
                if codepoint2 not in self.ngram_frequencies[language][codepoint1]:
                    self.ngram_frequencies[language][codepoint1][codepoint2] = dict()
                    self.ngram_frequencies[language][codepoint1][codepoint2][codepoint3] = 1
                else:
                    if codepoint3 not in self.ngram_frequencies[language][codepoint1][codepoint2]:
                        self.ngram_frequencies[language][codepoint1][codepoint2][codepoint3] = 1
                    else:
                        self.ngram_frequencies[language][codepoint1][codepoint2][codepoint3] += 1

    def get_ngram_probability(self, ngram, language):
        codepoint1 = ord(ngram[0])
        codepoint2 = ord(ngram[1])
        codepoint3 = ord(ngram[2])

        freq_ngram_for_language = self.smoothing_value
        try:
            freq_ngram_for_language += self.ngram_frequencies[language][codepoint1][codepoint2][codepoint3]
        except KeyError:
            pass

        total_ngram_count_for_language = self.ngram_frequencies[language]['total_count'] + \
                                         self.smoothing_value * (self.vocabulary_size ** 3)
        probability = freq_ngram_for_language / total_ngram_count_for_language

        if probability != 0:
            return log10(probability)
        else:
            return float('-inf')  # in case there is no smoothing value, assign negative infinity


class BYOMTrainingModel(TrainingModel):
    def parse_tweet(self, tweet):
        return self.trigramModel.parse_tweet(tweet)

    def __init__(self, vocabulary, smoothing_value, training_file):
        super().__init__(vocabulary, smoothing_value, training_file)

        self.unigramModel = UnigramTrainingModel(vocabulary, smoothing_value, training_file)
        self.bigramModel = BigramTrainingModel(vocabulary, smoothing_value, training_file)
        self.trigramModel = TrigramTrainingModel(vocabulary, smoothing_value, training_file)

    def train(self):
        self.unigramModel.train()
        self.bigramModel.train()
        self.trigramModel.train()

    def process_tweet(self, language, tweet):
        self.unigramModel.process_tweet(language, tweet)
        self.bigramModel.process_tweet(language, tweet)
        self.trigramModel.process_tweet(language, tweet)

        for bigram in self.bigramModel.parse_tweet(tweet):
            codepoint1 = ord(bigram[0])
            try:
                self.ngram_frequencies[language][codepoint1]['total_count'] += 1
            except KeyError:
                self.ngram_frequencies[language][codepoint1]['total_count'] = 1

        for trigram in self.trigramModel.parse_tweet(tweet):
            codepoint1 = ord(trigram[0])
            codepoint2 = ord(trigram[1])
            try:
                self.ngram_frequencies[language][codepoint1][codepoint2]['total_count'] += 1
            except KeyError:
                self.ngram_frequencies[language][codepoint1][codepoint2]['total_count'] = 1

    def get_ngram_probability(self, ngram, language):
        codepoint1 = ord(ngram[0])
        codepoint2 = ord(ngram[1])
        codepoint3 = ord(ngram[2])

        freq_unigram = self.smoothing_value
        freq_bigram = self.smoothing_value
        freq_trigram = self.smoothing_value

        try:
            freq_unigram += self.unigramModel.ngram_frequencies[language][codepoint1]
        except KeyError:
            pass

        try:
            freq_bigram += self.bigramModel.ngram_frequencies[language][codepoint1][codepoint2]
        except KeyError:
            pass

        try:
            freq_trigram += self.trigramModel.ngram_frequencies[language][codepoint1][codepoint2][codepoint3]
        except KeyError:
            pass

        freq_first_two_char_trigram = self.smoothing_value * (self.vocabulary_size ** 3)
        try:
            freq_first_two_char_trigram += self.trigramModel.ngram_frequencies[language][codepoint1][codepoint2][
                'total_count']
        except KeyError:
            pass

        freq_first_char_bigram = self.smoothing_value * (self.vocabulary_size ** 2)
        try:
            freq_first_char_bigram += self.bigramModel.ngram_frequencies[language][codepoint1]['total_count']
        except KeyError:
            pass

        unigram_count = self.smoothing_value * self.vocabulary_size
        try:
            unigram_count += self.unigramModel.ngram_frequencies[language]['total_count']
        except KeyError:
            pass

        try:
            p_trigram_given_first_two_chars = freq_trigram / freq_first_two_char_trigram
        except ZeroDivisionError:
            # in case there is no smoothing value, assign negative infinity
            return float('-inf')

        try:
            p_bigram_given_first_char = freq_bigram / freq_first_char_bigram
        except ZeroDivisionError:
            # in case there is no smoothing value, assign negative infinity
            return float('-inf')

        try:
            p_unigram = freq_unigram / unigram_count
        except ZeroDivisionError:
            # in case there is no smoothing value, assign negative infinity
            return float('-inf')

        return log10(p_trigram_given_first_two_chars) + log10(p_unigram) + log10(p_bigram_given_first_char)

    def get_probability_of_language(self, language):
        return self.unigramModel.get_probability_of_language(language)
