from learning_data import TokenUID
import glob
import datetime
categories = ['IT総合', '映画', '経済総合', '野球',
              '社会', 'ライフ総合', 'エンタメ総合', 'サッカー', 'スポーツ総合']
current_time = datetime.datetime.now()
output_name = current_time.strftime('%Y-%m-%d')


def dump_all_csv():
    tuid = TokenUID()
    csv_list = []
    for cat in categories:
        csv = glob.glob('csv/' + cat + '/*.csv')
        csv_list.extend(csv)

    tuid.update(csv_list)
    tuid.dump('tuid/' + output_name + '.tuid')


def update_tuid(prevfile):
    tuid = TokenUID()
    tuid.load(prevfile)
    loaded_date = datetime.datetime.strftime(prevfile, 'tuid/%Y-%m-%d.tuid')
    csv_list = []
    for cat in categories:
        date_format = 'YN_' + cat + '_%Y-%m-%d-%H-%M-%S.csv'
        dirname = 'csv/%s/' % (cat)
        csvs = glob.glob(dirname + '*.csv')
        for c in csvs:
            filename = c.replace(dirname, '')
            scrap_date = datetime.datetime.strptime(filename, date_format)
            if loaded_date < scrap_date:
                csv_list.extend(c)
    tuid.update(csv_list)
    tuid.dump('tuid/' + output_name + '.tuid')


dump_all_csv()
