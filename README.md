Repository URL: https://github.com/panoreak/comp472-nlp

**To Run:**

From the root directory run `python3 naive_bayes_classifier.py <V> <n> <δ> <training_file> <test_file>`

- `<V>` denotes the vocabulary to use:
    
    0 -> Case insensitive alphabet 
        (Only includes [a-z]. Chars in [A-Z] are converted to lowercase in both the training and test sets)
        
    1 -> Case sensitive alphabet
        (Includes [a-z] and [A-Z]. Lowercase and uppercase chars are considered separatedly)
        
    2 -> Characters that satisfy isalpha()
    
- `<n>` denotes the ngram size:
    
    1 -> Unigram
        
    2 -> Bigram
        
    3 -> Trigram

- `<δ>` denotes the smoothing value

- `<training_file>` denotes the name of the training file

- `<test_file>` denotes the name of the test file
