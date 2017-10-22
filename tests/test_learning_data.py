import unittest
import vectorize.learning_data as ld


class TestTokenUID(unittest.TestCase):

    def test_construct1(self):
        csv_list = ['tests/test1.csv']
        tuid = ld.Token()
        tuid.update(csv_list)
        self.assertEqual(tuid.token_seq_no, 4)
        self.assertEqual(tuid.loaded_csv_paths, csv_list)

    def test_construct2(self):
        csv_list = ['tests/test2.csv']
        tuid = ld.Token()
        tuid.update(csv_list)
        self.assertEqual(tuid.token_seq_no, 28)
        self.assertEqual(tuid.loaded_csv_paths, csv_list)
        self.assertEqual(tuid.token_to_id, {'プログラミング': 2, 'する': 9, '部分': 23, '最小限': 25, '汎用': 1, 'Python': 0, 'られる': 27, '設計': 8, '特徴': 18, 'シンプル': 5, '本体': 22, '抑える': 26,
                                            'やすい': 7, '少ない': 14, '核': 20, '比べる': 13, '必要': 24, 'れる': 10, '言語': 3, 'コード': 4, 'ある': 19, '数': 16, 'C': 12, 'なる': 21, '行': 15, 'いる': 11, '扱う': 6, '書ける': 17})

    def test_dump_and_load(self):
        filepath = 'tests/tokenuid.dump'
        csv_list = ['tests/test2.csv']
        tuid = ld.Token()
        tuid.update(csv_list)
        ld.dump(tuid, filepath)
        tuid2 = ld.load(filepath)
        self.assertEqual(tuid2.seq_no_uid, 28)
        self.assertEqual(tuid2.loaded_csv_list, csv_list)
        self.assertEqual(tuid2.token_dic, {'プログラミング': 2, 'する': 9, '部分': 23, '最小限': 25, '汎用': 1, 'Python': 0, 'られる': 27, '設計': 8, '特徴': 18, 'シンプル': 5, '本体': 22, '抑える': 26,
                                           'やすい': 7, '少ない': 14, '核': 20, '比べる': 13, '必要': 24, 'れる': 10, '言語': 3, 'コード': 4, 'ある': 19, '数': 16, 'C': 12, 'なる': 21, '行': 15, 'いる': 11, '扱う': 6, '書ける': 17})


class TestLearningData(unittest.TestCase):
    '''
    test class of mlearn.py LeaningData
    '''

    csv_list = ['tests/test2.csv']
    tuid = ld.Token()
    tuid.update(csv_list)
    ldata = ld.TfidfVectorizer(tuid)


if __name__ == '__main__':
    unittest.main()
