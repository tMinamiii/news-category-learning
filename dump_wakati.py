import glob
import json
import os

from vectorize.news_tokenizer import YahooNewsTokenizer

import constant_values as cv


def make_wakati(category):
    paths = glob.glob('json/{}/*.json'.format(category))
    tk = YahooNewsTokenizer()
    category_result = []
    for path in paths:
        with open(path, 'r') as f:
            chunk_list = json.load(f)
            for chunk in chunk_list:
                manuscript = chunk['manuscript']
                splitted = manuscript.split('\n')
                lines = [i for i in splitted if i.strip(' 　') != '']
                for line in lines:
                    sanitized = tk.sanitize(line)
                    tokens = tk.tokenize(sanitized)
                    line_wakati = ' '.join(tokens)
                    category_result.append(line_wakati)
    return category_result


def main():
    categories = cv.CATEGORIES
    for cat in categories:
        wakati = make_wakati(cat)
        if not os.path.isdir('wakati'):
            os.mkdir('wakati')
        filepath = 'wakati/{}.wakati'.format(cat)
        with open(filepath, mode='w') as f:
            f.write('\n'.join(wakati))


if __name__ == '__main__':
    main()
