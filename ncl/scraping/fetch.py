import csv
import datetime
import glob
import json
import os

from scraping import yahoonews as yahoonews
from vectorize.news_tokenizer import YahooNewsTokenizer


def write_news_file(filename, chunks, filetype):
    if filetype == 'json':
        if not os.path.isfile(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump([], f)

        with open(filename, 'r', encoding='utf-8') as f:
            feeds = json.load(f)

        with open(filename, 'w', encoding='utf-8') as f:
            feeds.extend(chunks)
            json.dump(feeds, f, ensure_ascii=False)

    elif filetype == 'csv':
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f,  lineterminator='\n')
            for chunk in chunks:
                writer.writerow([chunk['category'], chunk['title'],
                                 chunk['manuscript_len'], chunk['manuscript']])


def scrape(rss_dic, time, oneline=False) -> list:
    scraper = yahoonews.YahooNewsScraper()
    chunk_dic = {}
    for url in rss_dic.values():
        result = scraper.scrape_news(url, sleep=1, date=time, oneline=oneline)
        for k, v in result.items():
            if k in chunk_dic:
                chunk_dic[k].extend(v)
            else:
                chunk_dic[k] = v
    return chunk_dic


def fetch_news(rss_dic, time, filetype='json'):
    chunk_dic = scrape(rss_dic, time)
    for k, v in chunk_dic.items():
        timestr = time.strftime('%Y-%m-%d')
        targetdir = './data/json/{}'.format(timestr)
        if not os.path.isdir(targetdir):
            os.makedirs(targetdir)

        filename = '{0}/{1}.{2}'.format(targetdir, k, filetype)
        write_news_file(filename, v, filetype)


def fetch_wakati(time, filetype='json'):
    timestr = time.strftime('%Y-%m-%d')
    dirname = './data/wakati/{}'.format(timestr)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    paths = glob.glob('./data/{0}/{1}/*'.format(filetype, timestr))

    for path in paths:
        basename = os.path.basename(path)
        category = os.path.splitext(basename)[0]
        wakati = make_wakati(path)
        filename = './data/wakati/{0}/{1}.wakati'.format(timestr, category)
        write_wakati_file(filename, wakati)


def make_wakati(path):
    tk = YahooNewsTokenizer()
    category_result = []
    with open(path, 'r') as f:
        chunk_list = json.load(f)
        for chunk in chunk_list:
            manuscript = chunk['manuscript']
            splitted = manuscript.split('\n')
            lines = [i for i in splitted if i.strip(' 　') != '']
            for line in lines:
                sanitized = tk.sanitize(line)
                tokens = tk.tokenize(sanitized)
                line_wakati = ' '.join(tokens)
                category_result.append(line_wakati)
    return '\n'.join(category_result)


def write_wakati_file(filename, wakati):
    with open(filename, mode='w') as f:
        f.write(wakati)


def main(filetype):
    rss = yahoonews.YahooRSSScraper()

    jp = rss.scrape_jp_newslist()
    world = rss.scrape_world_newslist()
    economic = rss.scrape_economic_newslist()
    sports = rss.scrape_sports_newslist()
    it_science = rss.scrape_it_science_newslist()
    life = rss.scrape_life_newslist()
    entertaiment = rss.scrape_entertaiment_newslist()

    time = datetime.datetime.now()

    fetch_news(jp, time, filetype)
    fetch_news(world, time, filetype)
    fetch_news(economic, time, filetype)
    fetch_news(sports, time, filetype)
    fetch_news(it_science, time, filetype)
    fetch_news(life, time, filetype)
    fetch_news(entertaiment, time, filetype)

    fetch_wakati(time, filetype)
