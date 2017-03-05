from collections import Counter
import itertools
import math
import operator


class NaiveBayesClassifier:

    def __init__(self):
        self._classes = set()
        self._priors = {}
        self._likelihoods = {}

    def train(self, dataset):
        pass

    def classify(self, text):
        results = {}

        for class_id in self._classes:
            result = math.log(self._priors[class_id])

            for token in text:
                if token not in self._likelihoods:
                    continue

                result += math.log(self._likelihoods[token][class_id])

            results[class_id] = result

        class_id = max(results, key=results.get)

        return class_id
