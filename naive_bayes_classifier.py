import sys

from NGramClassifier import NGramClassifier

vocabulary = sys.argv[1]
ngram_size = sys.argv[2]
smoothing_value = sys.argv[3]
training_file = sys.argv[4]
test_file = sys.argv[5]

classifier = NGramClassifier(vocabulary, ngram_size, smoothing_value, training_file, test_file)
classifier.classify()
