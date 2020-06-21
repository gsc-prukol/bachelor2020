import keras
import pandas as pd
import json
import sys

print('Python %s on %s' % (sys.version, sys.platform))
from numpy import unicode
is_new = 0
if (is_new):
    from numpy import unicode

    file = open('review_category3.json')
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

# создадим массив, содержащий уникальные категории из нашего DataFrame
categories = {}
for key,value in enumerate(df['category'].unique()):
    categories[value] = key + 1


with open('categoryes_map.ison', 'w') as fp:
    json.dump(categories, fp)


# Запишем в новую колонку числовое обозначение категории
df['category_code'] = df['category'].map(categories)

total_categories = len(df['category'].unique()) + 1
print('Всего категорий: {}'.format(total_categories))

descriptions = df['Description']
categories = df['category_code']

# Посчитаем максимальную длинну текста описания в словах
max_words = 0
for desc in descriptions:
    words = len(desc.split())
    if words > max_words:
        max_words = words
print('Максимальная длина описания: {} слов'.format(max_words))

maxSequenceLength = max_words


from keras.preprocessing.text import Tokenizer

# создаем единый словарь (слово -> число) для преобразования
tokenizer = Tokenizer()
print(type(descriptions))
print(type(descriptions.tolist()[1]))
tokenizer.fit_on_texts(descriptions.tolist())

# Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
textSequences = tokenizer.texts_to_sequences(descriptions.tolist())

print ("замінені слова на числа")
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

X_train, y_train, X_test, y_test = load_data_from_arrays(textSequences, categories, train_test_split=0.85)

total_words = len(tokenizer.word_index)
print('В словаре {} слов'.format(total_words))

# количество наиболее часто используемых слов
num_words = total_words // 10

import numpy as np
num_classes = np.max(y_train) + 1

print(u'Преобразуем описания заявок в векторы чисел...')
tokenizer = Tokenizer(num_words=num_words)

with open('tokenizer.json', 'w') as f:
    t = tokenizer.to_json()
    print(t, file=f)

X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
print('Размерность X_train:', X_train.shape)
print('Размерность X_test:', X_test.shape)

print(u'Преобразуем категории в матрицу двоичных чисел '
      u'(для использования categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, Callback


class MyCustomCallback(Callback):
  def on_epoch_end(self, epoch, logs=None):
      with open('result.txt', 'a') as res:
          print(f'  Епоха = {epoch + 1}, ', file=res)
          print(f'  Оценка теста: {logs["val_loss"]}', file=res)
          print(f'  Оценка точности модели: {logs["val_accuracy"]}\n', file=res)


# количество эпох\итераций для обучения
epochs = 10
batch_size = 32
print(u'Собираем модель...')
model = Sequential()
model.add(Dense(int(1024 * 5), input_shape=(num_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(total_categories))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model = keras.models.load_model('./checkpoints/last_checkpoint.h5')
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)
model.summary()
import time
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks=[MyCustomCallback(), ModelCheckpoint(
                     f'./checkpoints/category_{time.localtime()}_{{epoch}}.h5')])

model.save_weights(f'./checkpoints/MPL_{epochs}_{len(X_train)}')

score = model.evaluate(X_test, y_test,
                       batch_size=32, verbose=1)
print()
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))


# import winsound
# winsound.PlaySound('sound.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)

with open ('result.txt', 'a') as res:
    print(f'Количество даных = {len(X_train)}, епох = {epochs}, размер пакета = {batch_size}', file=res)
    print(f'Оценка теста: {score[0]}', file=res)
    print(f'Оценка точности модели: {score[1]}', file=res)
    print("--------------------------------------------------------------------", file=res)
import matplotlib.pyplot as plt

# График точности модели
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# График оценки loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

with open('result_r.txt', 'a') as res:
    model.summary(print_fn= lambda x : print(x, file=res))
    print(f'Лучшая оценка теста: {np.max(history.history["val_loss"])}', file=res)
    print(f'Лучшая оценка точности модели: {np.max(history.history["val_accuracy"])}', file=res)
    print("""--------------------------------------------------------------------
====================================================================""", file=res)