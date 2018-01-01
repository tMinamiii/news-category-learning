import re

import mojimoji
from janome.tokenizer import Tokenizer


class YahooNewsTokenizer:
    def __init__(self):
        self._t = Tokenizer()

    def sanitize(self, manu: str) -> str:
        # 英文を取り除く（日本語の中の英字はそのまま）
        manu = re.sub(r'[a-zA-Z0-9]+[ \,\.\':;\-\+?!]', '', manu)
        # 記号や数字は「、」に変換する。
        # (単純に消してしまうと意味不明な長文になりjanomeがエラーを起こす)
        manu = re.sub(r'[0-9０-９]+', '0', manu)
        manu = re.sub(
            r'[!"#$%&()\*\+\-\.,\/:;<=>?@\[\\\]^_`{|}]+', '、', manu)
        manu = re.sub(r'[“（）【】『』｛｝「」［］《》〈〉]', '、', manu)
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
            if ps not in ['名詞', '動詞', '形容詞']:
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
