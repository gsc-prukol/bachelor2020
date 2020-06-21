import pandas as pd
import json
import sys
import pymorphy2
import re

print('Python %s on %s' % (sys.version, sys.platform))
from numpy import unicode

file = open('review_category2.json')
data = list()
for line in file:
    l = json.loads(line)
    data.append(l)

df = pd.DataFrame(data)
data = None
print('Завантаження даних')

def counter(func, counter_dict={}):
    counter_dict[func]=0
    def wrap(*args,**kwargs):
        counter_dict[func] += 1
        if counter_dict[func] % 1000 == 0:
            print(counter_dict[func])
        return func(*args,**kwargs)
    return wrap

ma = pymorphy2.MorphAnalyzer()

@counter
def clean_text(text):
    text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) #deleting newlines and line-breaks
    text = re.sub('[.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text) #deleting symbols
    text = " ".join(ma.parse(unicode(word))[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word)>3)

    return text


df['Description'] = df.apply(lambda x: clean_text(x['review']), axis=1)
df = df.drop(['review'], axis=1)

print('Створено очищений текст')

res_file = open('review_category2_clear.json', 'w')
df.to_json(res_file)
