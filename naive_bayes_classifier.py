import sys

from NGramClassifierFactory import NGramClassifierFactory

vocabulary = sys.argv[1]
ngram_size = sys.argv[2]
smoothing_value = sys.argv[3]
training_file = sys.argv[4]
test_file = sys.argv[5]

classifier = NGramClassifierFactory.get_ngram_classifier(vocabulary, ngram_size, smoothing_value, training_file,
                                                         test_file)
classifier.classify()
