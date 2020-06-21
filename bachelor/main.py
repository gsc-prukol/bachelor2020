import sqlite3

from keras.models import load_model
import pymorphy2
import re
import json
from pyproj import Geod
from numpy import unicode


ma = pymorphy2.MorphAnalyzer()

def _clean_text(text):
    text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols
    text = " ".join(ma.parse(unicode(word))[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word)>3)

    return text


from keras.preprocessing.text import Tokenizer, tokenizer_from_json
def _load_keras_model():
    """Load in the pre-trained model"""
    global model
    model = load_model('./checkpoints/epohs_10.h5')
    # Required for model to work
    # global graph
    # graph = tf.get_default_graph()
    global tokenizer
    with open('tokenizer.json') as f:
        tokenizer = tokenizer_from_json(f.readline())

    global category_map
    with open('categoryes_map.ison') as fp:
        category_tmp = json.load(fp)
        category_map = dict()
        for key in category_tmp:
            category_map[category_tmp[key]] = key

def _get_category(model,  tokenizer, text):
    # with graph.as_default():
        # Make a prediction from the seed
        text = _clean_text(text)
        text1 = tokenizer.texts_to_sequences([text])
        text1 = tokenizer.sequences_to_matrix(text1)
        preds = model.predict_classes(text1)

        return category_map[preds[0]]

def _get_restorans(latitude, longitude, categories, length = 10000, limit = 20):
    conn = sqlite3.connect("mydatabase.db")  # или :memory: чтобы сохранить в RAM
    print(f'open db, {latitude}, {longitude}, {categories}')
    cursor = conn.cursor()
    sql = f"SELECT * FROM business where categories=? and abs(latitude - ?) < {length / 100000} order by stars desc"
    data = []
    geod = Geod(ellps="WGS84")
    for row in cursor.execute(sql, [categories, latitude]):
        if geod.line_length((longitude, row[6]), (latitude, row[5])) < length:
            data.append(row)
    if len(data) > limit:
        return data[:limit]
    return data


_load_keras_model()

def get_restorans(text, latitude, longitude, length = 10000, limit = 20):
    category = _get_category(model=model, tokenizer=tokenizer, text=text)
    restorans = _get_restorans(latitude, longitude, category, length, limit)
    return restorans

print(get_restorans('I want tea', 36.173946, -115.154808, length=10000))