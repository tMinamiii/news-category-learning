import glob
import learning_data as ld
import datetime
import csv
import random
categories = ['IT総合', '映画', '経済総合', '野球',
              '社会', 'ライフ総合', 'エンタメ総合', 'サッカー', 'スポーツ総合']
#categories = ['IT総合']
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
svd_news = all_news[0:2000]
ldata = ld.LearningData()
svd_data = ldata.make(tuid, svd_news)
print('created training data for svd')
svd = ld.YN_SVD(svd_data[:, 1].tolist())
svd_ldata = ld.LearningData(svd)
svd_data = None
svd_news = None


def dump_all_csv():
    print('dumping tuid data...')
    ld.dump(tuid, 'ldata/' + output_name + '.tuid')
    print('making train data...')
    #td = ldata.make(tuid, all_news)
    td = svd_ldata.make(tuid, all_news)
    print('dumping train data...')
    #ld.dump(td, 'ldata/' + output_name + '.td')
    ld.dump(td, 'ldata/' + output_name + '.svdtd')


dump_all_csv()
