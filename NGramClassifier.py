from math import log10

from TrainingModelFactory import TrainingModelFactory
from VocabularyFactory import VocabularyFactory, CaseInsensitiveAlphabetChars
from Evaluation import Eval


class NGramClassifier:
    def __init__(self, vocabulary, ngram_size, smoothing_value, training_file, test_file):
        self.trace_file = 'trace_' + vocabulary + '_' + \
            ngram_size + '_' + smoothing_value + '.txt'
        self.eval_file = 'eval_' + vocabulary + '_' + \
            ngram_size + '_' + smoothing_value + '.txt'
        self.vocabulary = VocabularyFactory.get_vocabulary(vocabulary)

        # given δ must be within [0 ... 1]
        # if the given δ is smaller than 0, use 0
        # if the given δ is larger than 1, use 1
        self.smoothing_value = max(float(smoothing_value), 0)
        if self.smoothing_value is not 0:
            self.smoothing_value = min(float(smoothing_value), 1)

        self.training_model = TrainingModelFactory.get_training_model(self.vocabulary, ngram_size, self.smoothing_value,
                                                                      training_file)
        self.test_file = test_file

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

            ngrams = self.training_model.parse_tweet(tweet)
            highest_score = None
            language_with_highest_score = None
            for language in self.training_model.language_data.keys():
                score = 0
                for ngram in ngrams:
                    conditional_probability = self.training_model.get_ngram_probability(
                        ngram, language)
                    if conditional_probability != 0:
                        score += log10(conditional_probability)
                    else:
                        # in case there is no smoothing value, assign negative infinity
                        score += float('-inf')
                if highest_score is None or highest_score < score:
                    highest_score = score
                    language_with_highest_score = language

            languages_match = 'correct' if language_with_highest_score == actual_language else 'wrong'
            output_file.write(str.join('  ', [id, language_with_highest_score, str(highest_score), actual_language,
                                              languages_match]) + '\n')
        input_file.close()
        output_file.close()
        eval = Eval(self.trace_file, self.eval_file)
        eval.write_to_file()
