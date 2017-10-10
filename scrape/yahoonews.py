import xml.etree.ElementTree as ET
import urllib.request as req
import time
from bs4 import BeautifulSoup
from bs4.element import Tag
import datetime


class NewsChunk:
    def __init__(self, category, title, manuscript):
        self.category = category
        self.title = title
        self.manuscript = manuscript


class YahooNewsScraper:
    def read_manuscript(self, news_url) -> str:
        # readしてHTMLデータをすべてDLしてしまう
        resp = req.urlopen(news_url)
        code = resp.getcode()
        if code == 302:
            resp = req.urlopen(resp.geturl())
            html = resp.read()
        else:
            html = resp.read()
        soup = BeautifulSoup(html, 'lxml')
        paragraphs = soup.select('div.paragraph')
        manuscript = ''
        for paragraph in paragraphs:
            try:
                heading = paragraph.select_one('div.ynDetailHeading > em')
                if heading is not None:
                    manuscript += heading.string.strip(' 　')
                detail_txt = paragraph.select_one('p.ynDetailText')
                for con in detail_txt.contents:
                    if type(con) == Tag:
                        continue
                    manuscript += con.string.strip(' 　')
            except:
                print('Error occoured while scraping : ' + news_url)
        manuscript = manuscript.replace('\r', '')
        manuscript = manuscript.replace('\n', '')
        return manuscript

    def scrape_news(self, rss_url, sleep=1, date=None) -> dict:
        xml = req.urlopen(rss_url).read()
        root = ET.fromstring(xml)
        items = root.iter('item')
        news_dic = {}
        for item in items:
            if date is not None:
                pubdate_str = item.find('pubDate').text.strip()
                date_format = '%a, %d %b %Y %H:%M:%S %z'
                pubdate = datetime.datetime.strptime(pubdate_str, date_format)
                if pubdate.date() != date.date():
                    # rssなのでbreakでもよいが念の為
                    continue
            title = item.find('title').text.strip(' 　')
            link = item.find('link').text
            category = item.find('category').text
            manuscript = self.read_manuscript(link)
            chunk = NewsChunk(category, title, manuscript)
            if category in news_dic:
                news_dic[category].append(chunk)
            else:
                news_dic[category] = [chunk]
            time.sleep(sleep)
        return news_dic


class YahooRSSScraper:
    def __init__(self):
        rss_url = 'https://headlines.yahoo.co.jp/rss/list'
        html = req.urlopen(rss_url).read()
        soup = BeautifulSoup(html, 'lxml')
        news_areas = soup.select('div.rss_listbox')
        self.rss_dic = {}
        for area in news_areas:
            if area.select_one('h3').get('id') == 'news':
                # print(area)
                titles = area.select('div.ymuiHeaderBGLight > h4.ymuiTitle')
                containers = area.select('div.ymuiContainer')
                '''
                {'国内': 'JP', '国際': 'World', '経済': 'Economic',
                 'エンタメ': 'Entertaiment', 'スポーツ': 'Sports',
                 'IT・科学': 'Science', 'ライフ': 'Life', '地域': 'JPLocal'}
                '''
        for t_ml, con in zip(titles, containers):
            title = t_ml.contents[0]
            links = con.select('ul.ymuiList > li.ymuiArrow > dl')
            news_dic = {}
            for link in links[25:]:
                name = link.select_one('dt').string
                url = link.select_one('dd > a').get('href')
                news_dic.update({name: url})
            self.rss_dic.update({title: news_dic})

    def scrape_jp_newslist(self) -> dict:
        return self.rss_dic['国内']

    def scrape_world_newslist(self) -> dict:
        return self.rss_dic['国際']

    def scrape_economic_newslist(self)-> dict:
        return self.rss_dic['経済']

    def scrape_entertaiment_newslist(self)->dict:
        return self.rss_dic['エンタメ']

    def scrape_sports_newslist(self) -> dict:
        return self.rss_dic['スポーツ']

    def scrape_it_science_newslist(self) -> dict:
        return self.rss_dic['IT・科学']

    def scrape_life_newslist(self) -> dict:
        return self.rss_dic['ライフ']

    def scrape_jplocal_newslist(self)->dict:
        return self.rss_dic['地域']

    def scrape_all_newslist(self)->dict:
        return self.rss_dic
