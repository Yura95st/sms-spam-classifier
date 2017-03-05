import collections
import itertools
import functools
import math


class NaiveBayesClassifier:

    def __init__(self):
        self._classes = None
        self._priors = None
        self._likelihoods = None
        self._is_trained = False

    def train(self, dataset):
        dataset = dataset[:]

        self._init_classes(dataset)

        self._calculate_priors(len(dataset))

        self._calculate_likelihoods(dataset)

        self._is_trained = True

    def classify(self, text):
        if not self._is_trained:
            raise RuntimeError(
                "The classifer must be trained before classifying")

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

    def _init_classes(self, dataset):
        self._classes = collections.defaultdict(list)

        for class_id, text in dataset:
            self._classes[class_id].append(text)

    def _calculate_priors(self, total):
        self._priors = {}

        for class_id in self._classes:
            self._priors[class_id] = len(self._classes[class_id]) / total

    def _calculate_likelihoods(self, dataset):
        _, texts = zip(*dataset)

        vocabulary = set(functools.reduce(itertools.chain, texts, []))
        vocabulary_len = len(vocabulary)

        self._likelihoods = collections.defaultdict(dict)

        for class_id in self._classes:
            words = list(functools.reduce(
                itertools.chain, self._classes[class_id], []))

            words_len = len(words)
            words_counter = collections.Counter(words)

            for word in vocabulary:
                self._likelihoods[word][class_id] = (
                    words_counter[word] + 1) / (words_len + vocabulary_len)
