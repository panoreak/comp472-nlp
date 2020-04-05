from math import log10

from Evaluation import Eval
from TrainingModelFactory import TrainingModelFactory, BYOMTrainingModel
from VocabularyFactory import VocabularyFactory, CaseInsensitiveAlphabetChars


class Classifier:
    def __init__(self, training_file, test_file, byom, vocabulary='2', ngram_size='3', smoothing_value='0.001'):
        self.byom = byom
        self.vocabulary = VocabularyFactory.get_vocabulary(vocabulary)
        self.test_file = test_file

        # given δ must be within [0 ... 1]
        # if the given δ is smaller than 0, use 0
        # if the given δ is larger than 1, use 1
        self.smoothing_value = max(float(smoothing_value), 0)
        if self.smoothing_value is not 0:
            self.smoothing_value = min(float(smoothing_value), 1)

        if byom:
            self.trace_file = 'trace_myModel.txt'
            self.eval_file = 'eval_myModel.txt'
            self.training_model = BYOMTrainingModel(self.vocabulary, self.smoothing_value, training_file)

        else:
            self.trace_file = 'trace_' + vocabulary + '_' + ngram_size + '_' + smoothing_value + '.txt'
            self.eval_file = 'eval_' + vocabulary + '_' + ngram_size + '_' + smoothing_value + '.txt'
            self.training_model = TrainingModelFactory.get_nb_training_model(self.vocabulary, ngram_size,
                                                                             self.smoothing_value,
                                                                             training_file)

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
                score = log10(self.training_model.get_probability_of_language(language)) if self.byom else 0
                for ngram in ngrams:
                    probability = self.training_model.get_ngram_probability(ngram, language)
                    score += probability
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
