import sys

from learning import task
from scraping import fetch
from vectorize import tfidf_vectorizer, w2v_vectorizer
from vectorize import news_tokenizer
if __name__ == '__main__':
    command = sys.argv
    length = len(command)
    if length == 1:
        sys.exit()
    if command[1] == 'scraping':
        # csv or json
        filetype = command[2]
        print('scraping ...')
        fetch.main(filetype)
        print('scraping finished')
    elif command[1] == 'wakati':
        print('creating wakati files')
        # csv or json
        filetype = command[2]
        news_tokenizer.make_wakati(filetype)
        print('creating finished')
    elif command[1] == 'vectorize':
        # tfidf or word2vec
        vector_type = command[2]
        # csv or json
        filetype = command[3]
        if vector_type == 'word2vec':
            print('creating doc2vec models')
            w2v_vectorizer.main(filetype)
        elif vector_type == 'tfidf':
            print('creating tfidf vectors')
            tfidf_vectorizer.main(filetype)
        print('vectoring finished')
    elif command[1] == 'learning':
        # tfidf or word2vec
        learning_type = command[2]
        if learning_type == 'tfidf':
            task.main()
        elif learning_type == 'word2vec':
            pass
