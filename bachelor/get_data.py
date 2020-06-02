import json
#
# i = 1
# with open('dataset_business.json', encoding='ANSI') as bf:
#     with open('bisiness_category2.json', 'w') as bc:
#         dictb = {}
#         for line in bf:
#             b = json.loads(line)
#             if not b['categories']:
#                 continue
#             # categories = [x.strip().lower() for x in b['categories'].split(',')]
#             # dictb[b['business_id']] = categories
#             dictb[b['business_id']] = b['categories'].lower()
#         j = json.dumps(dictb)
#         print(j, file=bc)


# створення датасету меншої довжини
i = 1
with open('dataset_review.json', encoding='utf8') as bf:
    with open('dataset_review_short2.json', 'w') as bc:
        for line in bf:
            try:
                r = json.loads(line)
                j = json.dumps({'text': r['text'], 'business_id': r['business_id']})
                print(j, file=bc)
            except:
                continue
            if i > 10000:
                break
            if i % 1000 == 0:
                print(i)
            i += 1

# створення файлу відгук - категрія
i = 0
count = 0
with open('dataset_review_short2.json') as rev:
    with open('bisiness_category2.json') as bc:
        b = json.loads(bc.read())
        with open('review_category2.json', 'w') as rc:
            for line in rev:
                count += 1
                r = json.loads(line)
                business_id = r['business_id']
                categories = b.get(business_id)
                if not categories:
                    continue
                # if len(categories) == 1:
                # for c in categories:
                # c = categories[0]
                j = json.dumps({'review': r['text'].strip().lower(), 'category': categories})
                print(j, file=rc)
                i += 1
                # if count % 1000 == 0:
                #     print(f'count lines = {count}, i = {i}')

# i = 1
# with open('dataset_business.json', encoding='utf8') as bf:
#     with open('category.json', 'w') as bc:
#         classes = set()
#         for line in bf:
#             b = json.loads(line)
#             if not b['categories']:
#                 continue
#             categories = b['categories'].split(',')
#             for c in categories:
#                 j = c.strip().lower()
#                 classes |=  {j}
#                 i += 1
#         jr = json.dumps(list(classes))
#         print(jr, file=bc)
#         print(f'i = {i}')
#         print(f'j = {len(classes)}')


# import string, regex as re
#
# i = 1
# max_count = 0
# count = 0
# len_words = 0
# u_words = 0
# table = str.maketrans('', '', string.punctuation)
# with open('review_category.json') as bf:
#     dictionary = set()
#     for line in bf:
#         count += 1
#         b = json.loads(line)
#         reg = re.compile(r'[^\pL\p{Space}]')
#         review = reg.sub(' ', b['review'])
#         words = review.split()
#         words_tmp = {c.strip().lower() for c in words}
#         dictionary |= words_tmp
#         i += len(words)
#         len_words += len(words)
#         u_words += len(words_tmp)
#         if len(words) > max_count: max_count = len(words)
#         if count % 100000 == 0:
#             print(f'{count}: words {len_words}, u_tmp_words: {u_words}, u_g_words: {len(dictionary)}')
#             len_words = u_words = 0
#     jr = json.dumps(list(dictionary))
#     with open('dictionary.json', 'w') as bc:
#         print(jr, file=bc)
#     print(f'i = {i}')
#     print(f'j = {len(dictionary)}')


print('End!')
import winsound
winsound.PlaySound('sound.wav', winsound.SND_FILENAME)
