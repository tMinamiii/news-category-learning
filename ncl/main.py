import sys

from scraping import fetch
from vectorize import tfidf_vectorizer
from vectorize import w2v_vectorizer

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

    elif command[1] == 'integrate_wakati':
        print('integrate wakati files')
        fetch.integrate_wakati_files()

    elif command[1] == 'vectorize':
        vector_type = 'word2vec'
        if length == 3:
            vector_type = command[2]
        if vector_type == 'word2vec':
            w2v_vectorizer.main()
        elif vector_type == 'tfidf':
            tfidf_vectorizer.main()
