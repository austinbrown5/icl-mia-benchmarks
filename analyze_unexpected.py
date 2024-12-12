import json
import string
import difflib

with open('unexpected_words.json', 'r') as f:
    data = json.load(f)

#data is of len 869
#current exact match algo is wrong 43.5% of the time
# for entry in data:
entry = data[0]
pred = entry['unexpected_word']
options = entry['expected_words']

pred = pred.lower()
options = [option.lower().translate(str.maketrans('', '', string.punctuation)) for option in options]

entry2 = data[1]
pred2 = entry2['unexpected_word']
options2 = entry2['expected_words']

pred2 = pred2.lower()
options2 = [option.lower().translate(str.maketrans('', '', string.punctuation)) for option in options2]

for pred, options in [(pred[:-1], options), (pred2, options2)]:
    closest = difflib.get_close_matches(pred, options, n = 1, cutoff = 0.8)

missed = 0
for entry in data:
    pred = entry['unexpected_word']
    options = entry['expected_words']

    pred = pred.lower()
    options = [option.lower().translate(str.maketrans('', '', string.punctuation)) for option in options]
    closest = difflib.get_close_matches(pred, options, n = 1, cutoff = 0.7)
    if len(closest) == 0:
        missed += 1
        print(f'Miss {missed} \n')
        print(f'{pred}\n')
        print(f'{options}\n')

print(missed)



