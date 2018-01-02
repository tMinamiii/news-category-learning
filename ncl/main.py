import sys

from learning import task
from scraping import fetch
from vectorize import tfidf_vectorizer, w2v_vectorizer

if __name__ == '__main__':
    command = sys.argv
    length = len(command)
    if length == 1:
        sys.exit()
    if command[1] == 'scraping':
        filetype = 'json'
        if length == 3:
            filetype = command[2]
        print('scraping ...')
        fetch.main(filetype)

    elif command[1] == 'integrate_wakati':
        print('integrating wakati files')
        fetch.integrate_wakati_files()

    elif command[1] == 'vectorize':
        vector_type = 'word2vec'
        filetype = 'json'
        if length == 3:
            vector_type = command[2]
        elif length == 4:
            filetype = command[3]

        if vector_type == 'word2vec':
            print('creating doc2vec models')
            w2v_vectorizer.main()
        elif vector_type == 'tfidf':
            print('creating tfidf vectors')
            tfidf_vectorizer.main(filetype)
    elif command[1] == 'learning':
        learning_type = 'deep'
        if length == 3:
            learning_type = command[2]
        if learning_type == 'deep':
            task.main()
        elif learning_type == 'random_forrest':
            pass
