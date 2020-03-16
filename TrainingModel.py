from abc import ABC, abstractmethod

from VocabularyFactory import CaseInsensitiveAlphabetChars


class TrainingModel(ABC):

    def __init__(self, vocabulary, smoothing_value, training_file):
        self.vocabulary = vocabulary
        self.smoothing_value = smoothing_value
        self.training_file = training_file
        self.ngram_frequencies = dict()
        self.language_data = {
            'eu': {
                'doc_freq': 0,
                'total_ngram_count': 0
            },
            'ca': {
                'doc_freq': 0,
                'total_ngram_count': 0
            },
            'gl': {
                'doc_freq': 0,
                'total_ngram_count': 0
            },
            'es': {
                'doc_freq': 0,
                'total_ngram_count': 0
            },
            'en': {
                'doc_freq': 0,
                'total_ngram_count': 0
            },
            'pt': {
                'doc_freq': 0,
                'total_ngram_count': 0
            }
        }

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
    def get_conditional_probability(self, ngram, language):
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

        for codepoint in self.vocabulary.get_codepoint_list():
            self.ngram_frequencies[codepoint] = dict()
            for language in self.language_data.keys():
                # we add smoothing value to each ngram frequency upon initialization
                self.ngram_frequencies[codepoint][language] = self.smoothing_value

        # also add smoothing value times the size of the vocabulary to the total ngram count
        self.language_data[language]['total_ngram_count'] += self.smoothing_value * self.vocabulary.get_size()

    def process_tweet(self, language, tweet):
        for char in tweet:
            if self.vocabulary.is_in_vocabulary(char):
                self.ngram_frequencies[ord(char)][language] += 1
                self.language_data[language]['total_ngram_count'] += 1

    def get_conditional_probability(self, ngram, language):
        freq_ngram_for_language = self.ngram_frequencies[ord(ngram)][language]
        total_ngram_count_for_language = self.language_data[language]['total_ngram_count']
        return freq_ngram_for_language / total_ngram_count_for_language


class BigramTrainingModel(TrainingModel):
    def __init__(self, vocabulary, smoothing_value, training_file):
        super().__init__(vocabulary, smoothing_value, training_file)

    def process_tweet(self, language, tweet):
        pass

    def get_conditional_probability(self, ngram, language):
        pass


class TrigramTrainingModel(TrainingModel):
    def __init__(self, vocabulary, smoothing_value, training_file):
        super().__init__(vocabulary, smoothing_value, training_file)

    def process_tweet(self, language, tweet):
        pass

    def get_conditional_probability(self, ngram, language):
        pass
