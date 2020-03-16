from abc import ABC, abstractmethod
from math import log10

from TrainingModel import UnigramTrainingModel, BigramTrainingModel, TrigramTrainingModel
from VocabularyFactory import VocabularyFactory, CaseInsensitiveAlphabetChars


class NGramClassifierFactory:
    @staticmethod
    def get_ngram_classifier(vocabulary, ngram_size, smoothing_value, training_file, test_file):
        trace_file = 'trace_' + vocabulary + '_' + ngram_size + '_' + smoothing_value + '.txt'

        vocabulary = VocabularyFactory.get_vocabulary(vocabulary)

        # given δ must be within [0 ... 1]
        # if the given δ is smaller than 0, use 0
        # if the given δ is larger than 1, use 1
        smoothing_value = max(float(smoothing_value), 0)
        if smoothing_value is not 0:
            smoothing_value = min(float(smoothing_value), 1)

        if ngram_size == '1':
            training_model = UnigramTrainingModel(vocabulary, smoothing_value, training_file)
            return UnigramClassifier(vocabulary, smoothing_value, training_model, test_file, trace_file)

        if ngram_size == '2':
            training_model = BigramTrainingModel(vocabulary, smoothing_value, training_file)
            return BigramClassifier(vocabulary, smoothing_value, training_model, test_file, trace_file)

        if ngram_size == '3':
            training_model = TrigramTrainingModel(vocabulary, smoothing_value, training_file)
            return TrigramClassifier(vocabulary, smoothing_value, training_model, test_file, trace_file)


class NGramClassifier(ABC):
    def __init__(self, vocabulary, smoothing_value, training_model, test_file, trace_file):
        self.vocabulary = vocabulary
        self.smoothing_value = smoothing_value
        self.training_model = training_model
        self.test_file = test_file
        self.trace_file = trace_file

    def classify(self):
        self.training_model.train()
        self.test()

    def test(self):
        input_file = open(self.test_file, 'r', encoding="utf-8")
        output_file = open(self.trace_file, 'w', encoding="utf-8")
        for line in input_file:
            # skip empty lines
            if line is "\n":
                continue

            partitioned_line = line.split(maxsplit=3)
            id = partitioned_line[0]
            actual_language = partitioned_line[2]
            tweet = partitioned_line[3]

            if isinstance(self.vocabulary, CaseInsensitiveAlphabetChars):
                tweet = tweet.lower()

            ngrams = self.parse_tweet(tweet)
            highest_score = None
            language_with_highest_score = None
            for language in self.training_model.language_data.keys():
                score = log10(self.training_model.get_probability_of_language(language))
                for ngram in ngrams:
                    conditional_probability = self.training_model.get_conditional_probability(ngram, language)
                    if conditional_probability != 0:
                        score += log10(conditional_probability)
                    else:
                        score += float('-inf')  # in case there is no smoothing value, assign negative infinity
                if highest_score is None or highest_score < score:
                    highest_score = score
                    language_with_highest_score = language

            languages_match = 'correct' if language_with_highest_score == actual_language else 'wrong'
            output_file.write(str.join('  ', [id, language_with_highest_score, str(highest_score), actual_language,
                                              languages_match]) + '\n')

    # Returns list of ngrams found in tweet
    @abstractmethod
    def parse_tweet(self, tweet):
        pass


class UnigramClassifier(NGramClassifier):
    def __init__(self, vocabulary, smoothing_value, training_model, test_file, trace_file):
        super().__init__(vocabulary, smoothing_value, training_model, test_file, trace_file)

    def parse_tweet(self, tweet):
        unigrams = []
        for char in tweet:
            if self.vocabulary.is_in_vocabulary(char):
                unigrams.append(char)
        return unigrams


class BigramClassifier(NGramClassifier):
    def __init__(self, vocabulary, smoothing_value, training_model, test_file, trace_file):
        super().__init__(vocabulary, smoothing_value, training_model, test_file, trace_file)

    def parse_tweet(self, tweet):
        pass


class TrigramClassifier(NGramClassifier):
    def __init__(self, vocabulary, smoothing_value, training_model, test_file, trace_file):
        super().__init__(vocabulary, smoothing_value, training_model, test_file, trace_file)

    def parse_tweet(self, tweet):
        pass
