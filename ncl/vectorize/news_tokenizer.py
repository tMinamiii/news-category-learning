import glob
import os
import re
import shutil

from janome.tokenizer import Tokenizer

import mojimoji
import utils as u


class YahooNewsTokenizer:
    def __init__(self):
        self._t = Tokenizer(mmap=True)

    def sanitize(self, manu: str) -> str:
        # 英文を取り除く（日本語の中の英字はそのまま）
        manu = re.sub(r'[a-zA-Z0-9]+[ ,\.\'\:\;\-\+?\!]', '', manu)
        # 記号や数字は「 」に変換する。
        # (単純に消してしまうと意味不明な長文になりjanomeがエラーを起こす)
        manu = re.sub(r'[0-9０-９]+', '0', manu)
        manu = re.sub(r'[\!\?\#\$\%\&\'\*\+\-\.\^_\`\|\~\:\<\=\>\;\{\}\[\]\`\@\(\)\,\\]+',' ', manu)
        manu = re.sub(r'[“└┐（）【】『』｛｝「」［］《》〈〉！？＝]+', ' ', manu)
        return manu

    def tokenize(self, manuscript: str) -> list:
        token_list = []
        append = token_list.append
        try:
            tokens = self._t.tokenize(manuscript)
        except IndexError:
            print(manuscript)
            return None
        for tok in tokens:
            ps = tok.part_of_speech.split(',')[0]
            if ps not in ['名詞', '動詞', '形容詞', '副詞']:
                continue
            # 原形があれば原形をリストに入れる
            w = tok.base_form
            if w == '*' or w == '':
                # 原形がなければ表層系(原稿の単語そのまま)をリストに入れる
                w = tok.surface
            if w == '' or w == '\n':
                continue
            # 全角英数はすべて半角英数にする
            w = mojimoji.zen_to_han(w, kana=False, digit=False)
            # 半角カタカナはすべて全角にする
            w = mojimoji.han_to_zen(w, digit=False, ascii=False)
            # 英語はすべて小文字にする
            w = w.lower()
            append(w)
        return token_list


def make_wakati(filetype, clean=True):
    chunks = u.find_and_load_news(filetype)
    if clean:
        shutil.rmtree('./data/wakati')
    for ck in chunks:
        category = ck['category']
        ynt = YahooNewsTokenizer()
        sanitize = ynt.sanitize(ck['manuscript'])
        tokens = ynt.tokenize(sanitize)
        dirname = './data/wakati'
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        filepath = '{0}/{1}.wakati'.format(dirname, category)
        with open(filepath, 'a') as f:
            f.write(' '.join(tokens))
            f.write('\n')


def read_wakati():
    wakati_paths = glob.glob('./data/wakati/*.wakati')
    for path in wakati_paths:
        category = u.extract_category(path)
        print(category)
        with open(path, 'r') as f:
            all_wakati = f.read().split('\n')
        for line in all_wakati:
            tokens = line.split(' ')
            yield (category, tokens)
