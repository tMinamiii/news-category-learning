import sys

from scraping import fetch
from vectorize import vectorizer

if __name__ == '__main__':
    command = sys.argv
    length = len(command)
    print(length)
    if length == 1:
        sys.exit()
    if command[1] == 'scraping':
        filetype = 'json'
        if length == 3:
            filetype = command[2]
        print('scraping ...')
        fetch.main(filetype)
    if command[1] == 'vectorize':
        vector_type = 'word2vec'
        if length == 3:
            vector_type = command[2]

