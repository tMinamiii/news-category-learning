import sys

from ncl.learning import task
from ncl.vectorize import tfidf_vectorizer, w2v_vectorizer
from ncl.tokenizer import news_tokenizer


if __name__ == '__main__':
    command = sys.argv
    length = len(command)
    if length == 1:
        sys.exit()
    elif command[1] == 'tokenize':
        print('creating token files')
        news_tokenizer.make_tokenized_news()
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
    elif command[1] == 'deeplearning':
        # tfidf or word2vec
        task.main()
