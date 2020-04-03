from abc import ABC, abstractmethod

from VocabularyFactory import CaseInsensitiveAlphabetChars


class TrainingModelFactory:
    @staticmethod
    def get_training_model(vocabulary, ngram_size, smoothing_value, training_file):
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
                'ngram_count': 0,
            },
            'ca': {
                'doc_freq': 0,
                'ngram_count': 0,
            },
            'gl': {
                'doc_freq': 0,
                'ngram_count': 0,
            },
            'es': {
                'doc_freq': 0,
                'ngram_count': 0,
            },
            'en': {
                'doc_freq': 0,
                'ngram_count': 0,
            },
            'pt': {
                'doc_freq': 0,
                'ngram_count': 0,
            }
        }

        for language in self.language_data.keys():
            self.ngram_frequencies[language] = dict()

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
        unigrams = self.parse_tweet(tweet)
        self.language_data[language]['ngram_count'] += len(unigrams)
        for unigram in unigrams:
            codepoint = ord(unigram)

            if codepoint not in self.ngram_frequencies[language]:
                self.ngram_frequencies[language][ord(unigram)] = 1
            else:
                self.ngram_frequencies[language][ord(unigram)] += 1

    def get_ngram_probability(self, ngram, language):
        freq_ngram_for_language = self.smoothing_value
        try:
            freq_ngram_for_language += self.ngram_frequencies[language][ord(ngram)]
        except:
            pass

        total_ngram_count_for_language = self.language_data[language]['ngram_count'] + \
                                         self.smoothing_value * self.vocabulary_size
        return freq_ngram_for_language / total_ngram_count_for_language


class BigramTrainingModel(TrainingModel):
    def __init__(self, vocabulary, smoothing_value, training_file):
        super().__init__(vocabulary, smoothing_value, training_file)

    def process_tweet(self, language, tweet):
        pass

    def parse_tweet(self, tweet):
        pass

    def get_ngram_probability(self, ngram, language, vocabulary_size):
        pass


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
        trigrams = self.parse_tweet(tweet)
        self.language_data[language]['ngram_count'] += len(trigrams)
        for trigram in self.parse_tweet(tweet):
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
        freq_ngram_for_language = self.smoothing_value
        codepoint1 = ord(ngram[0])
        codepoint2 = ord(ngram[1])
        codepoint3 = ord(ngram[2])

        try:
            freq_ngram_for_language += self.ngram_frequencies[language][codepoint1][codepoint2][codepoint3]
        except:
            pass

        total_ngram_count_for_language = self.language_data[language]['ngram_count'] + \
                                         self.smoothing_value * (self.vocabulary_size ** 3)

        if total_ngram_count_for_language == 0:
            return 0
        return freq_ngram_for_language / total_ngram_count_for_language
