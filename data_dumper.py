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
for path in csv_list:
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            all_news.append(row)
print('all news loaded')
random.shuffle(all_news)
pca_news = all_news[0:1499]
ldata = ld.LearningData()
pca_data = ldata.make(tuid, pca_news)
print('created training data for pca')
pca = ld.YN_PCA(pca_data[:, 1].tolist())
pca_ldata = ld.LearningData(pca)
pca_data = None
pca_news = None


def dump_all_csv():
    ld.dump(tuid, 'ldata/' + output_name + '.tuid')
    #td = ldata.make(tuid, all_news)
    #ld.dump(td, 'ldata/' + output_name + '.td')
    td = pca_ldata.make(tuid, all_news)
    ld.dump(td, 'ldata/' + output_name + '.pcatd')


def update_tuid(prevfile):
    prev_tuid = ld.load(prevfile)
    loaded_date = datetime.datetime.strftime(prevfile, 'tuid/%Y-%m-%d.tuid')
    newly_added_csv = []
    for cate in categories:
        date_format = 'YN_' + cate + '_%Y-%m-%d-%H-%M-%S.csv'
        dirname = 'csv/%s/' % (cate)
        csvs = glob.glob(dirname + '*.csv')
        for csv_path in csvs:
            filename = csv_path.replace(dirname, '')
            scrap_date = datetime.datetime.strptime(filename, date_format)
            if loaded_date < scrap_date:
                newly_added_csv.extend(csv_path)
    prev_tuid.update(csv_list)
    ld.dump(tuid, 'tuid/' + output_name + '.tuid')
    td = ldata.make(tuid, all_news)
    ld.dump(td, 'ldata/' + output_name + '.td')


dump_all_csv()
