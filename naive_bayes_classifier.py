import sys

from NGramClassifier import NGramClassifier

classifier = NGramClassifier(sys.argv)
classifier.classify()
