import json
import pprint

exist = { 'meat shops',
         'bagels', 'specialty food'}
def isExist(text):
    for e in exist:
        if e in text:
            return True
    return False

def get_class(text):
    if 'american' in text:
        return 'american'
    elif 'wine' in text:
        return 'wine'
    elif 'vegan' in text or 'vegetarian' in text:
        return 'vegan'
    elif 'tea' in text or 'coffee' in text:
        return 'tea & coffee'
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
    return None


i = 0
count = 0
dictm = {}
with open('bisiness_category2.json') as bc:
    b = json.loads(bc.read())
    for bisn in b:
        if 'food' in b[bisn]:
            categ = get_class(b[bisn])
            if not categ is None:
                dictm[bisn] = get_class(b[bisn])

jr = json.dumps(dictm)
with open('bisiness_category3.json', 'w') as bc:
    print(jr, file=bc)
