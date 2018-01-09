import datetime
import sys

from learning import task
from scraping import fetch
from vectorize import news_tokenizer, tfidf_vectorizer, w2v_vectorizer

if __name__ == '__main__':
    command = sys.argv
    length = len(command)
    if length == 1:
        sys.exit()
    if command[1] == 'scraping':
        # csv or json
        filetype = command[2]
        print('scraping ...')
        now = datetime.datetime.now()
        fetch.main(filetype, time=now)
        news_tokenizer.make_wakati(filetype, time=now)
        print('scraping finished')
    elif command[1] == 'wakati':
        print('creating wakati files')
        # csv or json
        filetype = command[2]
        news_tokenizer.make_wakati(filetype, clean=True)
        print('creating finished')
    elif command[1] == 'vectorize':
        # tfidf or word2vec
        vector_type = command[2]
        if vector_type == 'word2vec':
            print('creating doc2vec models')
            validation = command[3]
            if validation == 'validation':
                w2v_vectorizer.main(validation=True)
            elif validation == 'make_model':
                w2v_vectorizer.main(validation=False)
        elif vector_type == 'tfidf':
            print('creating tfidf vectors')
            tfidf_vectorizer.main()
        print('vectoring finished')
    elif command[1] == 'learning':
        # tfidf or word2vec
        learning_type = command[2]
        if learning_type == 'tfidf':
            task.main()
        elif learning_type == 'word2vec':
            pass
