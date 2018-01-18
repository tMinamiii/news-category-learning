import ftplib
import glob
import posixpath
import re
import tempfile
from ftplib import error_perm
from posixpath import dirname

import MeCab

import mojimoji
from ncl import settings, utils


class YahooNewsTokenizer:
    def __init__(self):
        self._m = MeCab.Tagger(' -d  /usr/lib/mecab/dic/mecab-ipadic-neologd')
        # compileしておく
        self.eng_sentences = re.compile(r'[a-zA-Z0-9]+[ ,\.\'\:\;\-\+?\!]')
        self.numbers = re.compile(r'[0-9０-９]+')
        self.symbols1 = re.compile(r'[\!\?\#\$\%\&\'\*\+\-\.\^_\`\|\~\:]+')
        self.symbols2 = re.compile(r'[\<\=\>\;\{\}\[\]\`\@\(\)\,\\]+')
        self.cjk_symbols = re.compile(r'[“└┐（）【】『』｛｝「」［］《》〈〉！？＝]+')

    def sanitize(self, manu: str) -> str:
        # 英文を取り除く（日本語の中の英字はそのまま）
        manu = re.sub(self.eng_sentences, '', manu)
        # 記号や数字は「 」に変換する。
        # (単純に消してしまうと意味不明な長文になりjanomeがエラーを起こす)
        manu = re.sub(self.numbers, '0', manu)
        manu = re.sub(self.symbols1, ' ', manu)
        manu = re.sub(self.symbols2, ' ', manu)
        manu = re.sub(self.cjk_symbols, ' ', manu)
        return manu

    def tokenize(self, manuscript: str) -> list:
        token_list = []
        append = token_list.append
        try:
            tokens = self._m.parse(manuscript).split('\n')
        except IndexError:
            print(manuscript)
            return None
        for tok in tokens:
            # 表層形\t品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用形,活用型,原形,読み,発音
            tok = re.split(r'[\,\t]', tok)
            if len(tok) < 10:
                continue
            ps = tok[1]
            if ps not in ['名詞', '動詞', '形容詞']:
                continue
            # 原形があれば原形をリストに入れる
            w = tok[7]
            if w == '*' or w == '':
                # 原形がなければ表層系(原稿の単語そのまま)をリストに入れる
                w = tok[0]
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


def make_tokenized_news():
    all_chunks = utils.find_and_load_ftp_files()
    ftp = ftplib.FTP()
    ftp.encoding = 'utf-8'
    ftp.connect(settings.FTP_SERVER, 21)
    ftp.login(settings.FTP_USER, settings.FTP_PASS)
    for filename, chunks in all_chunks.items():
        with tempfile.TemporaryFile() as fp:
            for chunk in chunks:
                category = chunk['category']
                ynt = YahooNewsTokenizer()
                sanitize = ynt.sanitize(chunk['manuscript'])
                tokens = ynt.tokenize(sanitize)
                line = ' '.join(tokens) + '\n'
                fp.write(line.encode('utf-8'))
            filepath = '/{0}/{1}/{2}.token'.format(
                settings.FTP_TOKEN_DIR,
                category,
                filename)
            dirname, filename = posixpath.split(filepath)
            ftp_makedirs_cwd(ftp, dirname)
            fp.seek(0)
            ftp.storbinary('STOR %s' % filepath, fp)
    ftp.quit()


def ftp_makedirs_cwd(ftp, path, first_call=True):
    """Set the current directory of the FTP connection given in the `ftp`
    argument (as a ftplib.FTP object), creating all parent directories if they
    don't exist. The ftplib.FTP object must be already connected and logged in.
    """
    try:
        ftp.cwd(path)
    except error_perm:
        ftp_makedirs_cwd(ftp, dirname(path), False)
        ftp.mkd(path)
        if first_call:
            ftp.cwd(path)


def read_tokenized_news():
    wakati_paths = glob.glob('./data/token/*.token')
    print('reading token files')
    for path in wakati_paths:
        category = utils.extract_category(path)
        with open(path, 'r') as f:
            all_wakati = f.read().split('\n')
        for line in all_wakati:
            tokens = line.split(' ')
            yield (category, tokens)
