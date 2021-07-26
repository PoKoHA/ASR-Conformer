import json

def load_label_json(labels_path):
    with open(labels_path, encoding='utf-8') as label_file:
        labels = json.load(label_file)
        char2index = dict()
        index2char = dict()

        for index, char in enumerate(labels):
            char2index[char] = index
            index2char[index] = char

        return char2index, index2char  # todo labels 형태


def load_label_index(label_path):
    char2index = dict()
    index2char = dict()
    print(label_path)

    with open(label_path, 'r', encoding='utf-8') as f:
        for no, line in enumerate(f):
            if line[0] == '#':
                continue

            index, char, freq = line.strip().split('\t')  # strip 양쪽 공백 제거
            char = char.strip()

            if len(char) == 0:
                char = ' '

            char2index[char] = int(index)
            index2char[int(index)] = char

    return char2index, index2char