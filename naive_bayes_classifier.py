import sys

from Classifier import Classifier

classifier = None

if sys.argv[1] == 'byom':
    training_file = sys.argv[2]
    test_file = sys.argv[3]
    classifier = Classifier(training_file, test_file, True)
else:
    vocabulary = sys.argv[1]
    ngram_size = sys.argv[2]
    smoothing_value = sys.argv[3]
    training_file = sys.argv[4]
    test_file = sys.argv[5]
    classifier = Classifier(training_file, test_file, False, vocabulary, ngram_size, smoothing_value)

classifier.classify()
