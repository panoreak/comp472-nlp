from VocabularyFactory import VocabularyFactory
import collections


class NGramClassifier:
    def __init__(self, argv):
        self.vocabulary = VocabularyFactory.get_vocabulary(argv[1])
        self.ngram_size = argv[2]

        # given δ must be within [0 ... 1]
        # if the given δ is smaller than 0, use 0
        # if the given δ is larger than 1, use 1
        self.smoothing_value = max(float(argv[3]), 0)
        if self.smoothing_value is not 0:
            self.smoothing_value = min(float(argv[3]), 1)

        self.training_file = argv[4]
        self.test_file = argv[5]

        self.training_model = TrainingModel(self.vocabulary, self.ngram_size, self.smoothing_value, self.training_file)

    def test(self):
        pass

    def classify(self):
        self.training_model.train()
        self.test()


class TrainingModel:
    def __init__(self, vocabulary, ngram_size, smoothing_value, training_file):
        self.vocabulary = vocabulary
        self.ngram_size = ngram_size
        self.smoothing_value = smoothing_value
        self.training_file = training_file

        self.training_data = {
            'eu': {
                'freq': 0
            },
            'ca': {
                'freq': 0
            },
            'gl': {
                'freq': 0
            },
            'es': {
                'freq': 0
            },
            'en': {
                'freq': 0
            },
            'pt': {
                'freq': 0
            }
        }

    def train(self):
        input_file = open(self.training_file, 'r', encoding="utf-8")
        for line in input_file:
            partitioned_line = line.split(maxsplit=3)
            language = partitioned_line[2]
            tweet = partitioned_line[3]

            self.training_data[language]['freq'] += 1
            for char in tweet:
                codepoint = ord(char)

    def get_num_docs(self):
        sum = 0
        for language_data in self.training_data.values():
            sum += language_data['freq']
        return sum
