import json
import sqlite3

def get_class(text):
    text = text.lower()
    if 'american' in text:
        return 'american'
    elif 'wine' in text:
        return 'wine'
    elif 'vegan' in text or 'vegetarian' in text:
        return 'vegan'
    elif 'pizza' in text:
        return 'pizza'
    elif 'fast food' in text:
        return 'fast food'
    elif 'chinese' in text:
        return 'chinese'
    elif 'mongolian' in text:
        return 'mongolian'
    elif 'japanese' in text:
        return 'japanese'
    elif 'korean' in text:
        return 'korean'
    elif 'thai' in text:
        return 'thai'
    elif 'indian' in text:
        return 'indian'
    elif 'italian' in text:
        return 'italian'
    elif 'austrian' in text:
        return 'austrian'
    elif 'greek' in text:
        return 'greek'
    elif 'mexican' in text:
        return 'mexican'
    elif 'canadian' in text:
        return 'canadian'
    elif 'greek' in text:
        return 'greek'
    elif 'middle eastern' in text:
        return 'middle eastern'
    elif 'african' in text:
        return 'african'
    elif 'ethnic food' in text:
        return 'ethnic food'
    elif 'beer' in text:
        return 'beer'
    elif 'bar' in text:
        return 'bar'
    elif 'hot dog' in text:
        return 'hot dog'
    elif 'hot pot' in text:
        return 'hot pot'
    elif 'bakeries' in text:
        return 'bakeries'
    elif 'vitamins' in text or 'fruits' in text or 'veggies' in text:
        return 'vitamins'
    elif 'seafood' in text:
        return 'seafood'
    elif 'burgers' in text:
        return 'burgers'
    elif 'sandwiches' in text:
        return 'sandwiches'
    elif 'chocolatiers' in text or 'ice cream' in text or 'desserts' in text or 'bagels' in text or 'candy' in text :
        return 'desserts'
    elif 'donuts' in text:
        return 'donuts'
    elif 'farmers' in text:
        return 'farmers'
    elif 'meat' in text:
        return 'meat'
    elif 'specialty' in text:
        return 'specialty'
    elif 'tea' in text or 'coffee' in text:
        return 'tea & coffee'
    return None


conn = sqlite3.connect("mydatabase.db")  # или :memory: чтобы сохранить в RAM
cursor = conn.cursor()

# Создание таблицы
cursor.execute('drop table business')
cursor.execute("""CREATE TABLE business
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name text, address text, city text,
                  state text, latitude real, longitude real,
                  stars real, categories text)
               """)
i = 1
data = []
with open('dataset_business.json', encoding='utf8') as bf:
    for line in bf:
        try:
            r = json.loads(line)
            if r['categories'] and get_class(r['categories']):
                print(r['categories'], get_class(r['categories']))
                data.append((r['name'], r['address'], r['city'], r['state'], r['latitude'], r['longitude'], r['stars'],
                            get_class(r['categories'])))
        except():
            continue
        if i % 10000 == 0:
            print(i)
        i += 1

cursor.executemany('INSERT INTO business (name, address, city, state, latitude, longitude, stars, categories) \n'
                   'VALUES (?, ?, ?, ?, ?, ?, ?, ?)', data)
# Сохраняем изменения
conn.commit()
# for row in cursor.execute('select * from business where categories is not null'):
#     print(row)



