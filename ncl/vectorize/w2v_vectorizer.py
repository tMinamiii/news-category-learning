from gensim.models import word2vec
import glob
import os


def main():
    dirname = './data/vector/w2v'
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    read_paths = glob.glob('./data/wakati/*.wakati')
    for path in read_paths:
        basename = os.path.basename(path)
        category, _ = os.path.splitext(basename)

        data = word2vec.LineSentence(path)
        model = word2vec.Word2Vec(
            data, size=100, window=5, hs=1, min_count=1, sg=1)
        model.save('./data/vector/w2v/{}.model'.format(category))
