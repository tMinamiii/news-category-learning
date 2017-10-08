import glob
import learning_data as ld
import datetime
import csv
import random
categories = ['IT総合', '映画', '経済総合', '野球',
              '社会', 'ライフ総合', 'エンタメ総合', 'サッカー', 'スポーツ総合']
current_time = datetime.datetime.now()
output_name = current_time.strftime('%Y-%m-%d')
csv_list = []
for cat in categories:
    path = glob.glob('csv/' + cat + '/*.csv')
    csv_list.extend(path)
tuid = ld.TokenUID()
tuid.update(csv_list)

# dump_all_csv()
all_news = []
for csv_path in csv_list:
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            all_news.append(row)
all_news = random.shuffle(all_news)
pca_news = all_news[0:999]
ldata = ld.LearningData()
pca_data = ldata.make(tuid, pca_news)
pca = ld.YN_PCA(pca_data)
pca_ldata = ld.LearningData(pca)
pca_data = None


def dump_all_csv():
    tuid = ld.TokenUID()
    tuid.update(csv_list)
    ld.dump(tuid, 'ldata/' + output_name + '.tuid')
    #td = ldata.make(tuid, all_news)
    #ld.dump(td, 'ldata/' + output_name + '.td')
    td = pca_ldata.make(tuid, all_news)
    ld.dump(td, 'ldata/' + output_name + '.pcatd')


def update_tuid(prevfile):
    tuid = ld.load(prevfile)
    loaded_date = datetime.datetime.strftime(prevfile, 'tuid/%Y-%m-%d.tuid')
    for cate in categories:
        date_format = 'YN_' + cate + '_%Y-%m-%d-%H-%M-%S.csv'
        dirname = 'csv/%s/' % (cate)
        csvs = glob.glob(dirname + '*.csv')
        for c in csvs:
            filename = c.replace(dirname, '')
            scrap_date = datetime.datetime.strptime(filename, date_format)
            if loaded_date < scrap_date:
                csv_list.extend(c)
    tuid.update(csv_list)
    ld.dump(tuid, 'tuid/' + output_name + '.tuid')
    td = ldata.make(tuid, all_news)
    ld.dump(td, 'ldata/' + output_name + '.td')


dump_all_csv
