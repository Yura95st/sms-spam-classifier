from message_type import MessageType
import string


def read_data(file_path):
    translator = str.maketrans({p: ' ' for p in string.punctuation})

    with open(file_path, 'r') as data_file:
        for line in data_file:
            line = line.translate(translator).split()

            msg_type = MessageType.ham if line[
                0] == 'ham' else MessageType.spam
            msg_text = line[1:]

            yield msg_type, msg_text


def main(file_path='data/SMSSpamCollection'):
    messages = []

    for msg in read_data(file_path):
        messages.append(msg)


if __name__ == '__main__':
    main()
