from __future__ import print_function

import pandas as pd
import json

is_new = False
if (is_new):
    from numpy import unicode

    file = open('review_category.json')
    data = list()
    for line in file:
        l = json.loads(line)
        data.append(l)

    df = pd.DataFrame(data)

    print('Завантаження даних')

    import pymorphy2
    import re

    ma = pymorphy2.MorphAnalyzer()

    def clean_text(text):
        text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
        text = text.lower()
        text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
        text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols
        text = " ".join(ma.parse(unicode(word))[0].normal_form for word in text.split())
        text = ' '.join(word for word in text.split() if len(word)>3)

        return text

    df['Description'] = df.apply(lambda x: clean_text(x['review']), axis=1)

    print('Створено очищений текст')

    res_file = open('review_category_clear.json', 'w')
    df.to_json(res_file)
else:
    with open('review_category_clear.json', 'r') as res_file:
        df = pd.read_json(res_file)


def load_data_from_arrays(strings, labels, train_test_split=0.9):
    data_size = len(strings)
    test_size = int(data_size - round(data_size * train_test_split))
    print("Test size: {}".format(test_size))

    print("\nTraining set:")
    x_train = strings[test_size:]
    print("\t - x_train: {}".format(len(x_train)))
    y_train = labels[test_size:]
    print("\t - y_train: {}".format(len(y_train)))

    print("\nTesting set:")
    x_test = strings[:test_size]
    print("\t - x_test: {}".format(len(x_test)))
    y_test = labels[:test_size]
    print("\t - y_test: {}".format(len(y_test)))

    return x_train, y_train, x_test, y_test

# создадим массив, содержащий уникальные категории из нашего DataFrame

descriptions = df['Description']


# Максимальное количество слов в самом длинном описании заявки
max_words = 0
for desc in descriptions.tolist():
    words = len(desc.split())
    if words > max_words:
        max_words = words
print('Максимальное количество слов в самом длинном описании заявки: {} слов'.format(max_words))




import logging
import multiprocessing
import gensim
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Считываем файл с предобработанными твитами
data = df['Description']
# Обучаем модель
print(data[0])
model = Word2Vec(data, size=200, window=5, min_count=3, workers=multiprocessing.cpu_count())
model.
model.save("models/model.h5", separately=list('\n'))