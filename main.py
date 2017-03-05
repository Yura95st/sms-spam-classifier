from message_type import MessageType
from dataset_utils import DatasetUtils
from naive_bayes_classifier import NaiveBayesClassifier


def main(filename='data/SMSSpamCollection', n_folds=5):
    dataset = DatasetUtils.load_from_file(filename)
    dataset_folds = DatasetUtils.cross_validation_split(dataset, n_folds)

    classifier = NaiveBayesClassifier()

    results = []

    for test_set in dataset_folds:
        training_set = dataset_folds[:]
        training_set.remove(test_set)
        training_set = sum(training_set, [])

        classifier.train(training_set)

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for msg_type, msg_text in test_set:
            classified_type = classifier.classify(msg_text)

            if msg_type == MessageType.ham:
                if classified_type == MessageType.ham:
                    true_negative += 1
                else:
                    false_positive += 1

            else:
                if classified_type == MessageType.spam:
                    true_positive += 1
                else:
                    false_negative += 1

        print("True positives: {}".format(true_positive))
        print("True negatives: {}".format(true_negative))
        print("False positives: {}".format(false_positive))
        print("False negatives: {}".format(false_negative))
        print()

        accuracy = (true_positive + true_negative) / len(test_set)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f_one = (2 * precision * recall) / (precision + recall)

        print("Accuracy: {}".format(accuracy))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f_one))
        print()

        results.append((accuracy, precision, recall, f_one))

    mean_accuracy = sum([r[0] for r in results]) / len(results)
    print("Mean accuracy: {}".format(mean_accuracy))


if __name__ == '__main__':
    main()
