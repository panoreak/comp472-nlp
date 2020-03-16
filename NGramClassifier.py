from VocabularyFactory import VocabularyFactory


class NGramClassifier:
    def __init__(self, argv):
        self.vocabulary = VocabularyFactory.get_vocabulary(argv[1])
        self.vocabulary.is_in_vocabulary('a')
        self.ngram_size = argv[2]
        self.smoothing_value = argv[3]
        self.training_file = argv[4]
        self.test_file = argv[5]

    def train(self):
        print(self.training_file)

    def test(self):
        print(self.test_file)

    def classify(self):
        self.train()
        self.test()
