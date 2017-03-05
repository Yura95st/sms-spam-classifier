from message_type import MessageType
import random
import string


class DatasetUtils:

    @staticmethod
    def load_from_file(filename):
        dataset = []
        translator = str.maketrans({p: ' ' for p in string.punctuation})

        with open(filename, 'r') as data_file:
            for line in data_file:
                line = line.translate(translator).split()

                msg_type = MessageType.ham if line[
                    0] == 'ham' else MessageType.spam
                msg_text = line[1:]

                dataset.append((msg_type, msg_text))
        
        return dataset

    @staticmethod
    def cross_validation_split(dataset, n_folds):
        folds = []
        dataset_copy = dataset[:]
        fold_size = round(len(dataset_copy) / n_folds)

        random.shuffle(dataset_copy)

        for _ in range(n_folds - 1):
            fold = dataset_copy[:fold_size]

            dataset_copy = dataset_copy[fold_size:]
            folds.append(fold)

        folds.append(dataset_copy)

        return folds
