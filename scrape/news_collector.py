import csv
import datetime
import os

import scrape.yahoonews


def dump_csv(filename, chunk_list: list):
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f,  lineterminator='\n')
        for chunk in chunk_list:
            writer.writerow([chunk.category, chunk.title, len(
                chunk.manuscript), chunk.manuscript])


def scrape(rss_dic: dict, date: datetime) -> list:
    scraper = yahoonews.YahooNewsScraper()
    chunk_dic = {}
    for url in rss_dic.values():
        result = scraper.scrape_news(url, sleep=2, date=date)
        for k, v in result.items():
            if k in chunk_dic:
                chunk_dic[k].extend(v)
            else:
                chunk_dic[k] = v
    return chunk_dic


def collect(rss_dic: dict, current_time: datetime):
    chunk_dic = scrape(rss_dic, time)
    timestr = current_time.strftime('%Y-%m-%d')  # -%H-%M-%S')
    for k, v in chunk_dic.items():
        targetdir = 'csv/' + k
        if not os.path.isdir(targetdir):
            os.makedirs(targetdir)

        filename = targetdir + '/YN_' + k + '_' + timestr + '.csv'
        dump_csv(filename, v)


if __name__ == '__main__':
    rss = yahoonews.YahooRSSScraper()

    jp = rss.scrape_jp_newslist()
    world = rss.scrape_world_newslist()
    economic = rss.scrape_economic_newslist()
    sports = rss.scrape_sports_newslist()
    it_science = rss.scrape_it_science_newslist()
    life = rss.scrape_life_newslist()
    entertaiment = rss.scrape_entertaiment_newslist()

    time = datetime.datetime.now()
    collect(jp, time)
    collect(world, time)
    collect(economic, time)
    collect(sports, time)
    collect(it_science, time)
    collect(life, time)
    collect(entertaiment, time)
