import unittest
import numpy as np
import learning_data as ld


class TestTokenUID(unittest.TestCase):

    def test_construct1(self):
        csv_list = ['tests/test1.csv']
        tuid = ld.TokenUID()
        tuid.update(csv_list)
        self.assertEqual(tuid.seq_no_uid, 4)
        self.assertEqual(tuid.loaded_csv_list, csv_list)
        self.assertEqual(tuid.token_dic, {
                         '汎用': 1, '言語': 3, 'Python': 0, 'プログラミング': 2})

    def test_construct2(self):
        csv_list = ['tests/test2.csv']
        tuid = ld.TokenUID()
        tuid.update(csv_list)
        self.assertEqual(tuid.seq_no_uid, 28)
        self.assertEqual(tuid.loaded_csv_list, csv_list)
        self.assertEqual(tuid.token_dic, {'プログラミング': 2, 'する': 9, '部分': 23, '最小限': 25, '汎用': 1, 'Python': 0, 'られる': 27, '設計': 8, '特徴': 18, 'シンプル': 5, '本体': 22, '抑える': 26,
                                          'やすい': 7, '少ない': 14, '核': 20, '比べる': 13, '必要': 24, 'れる': 10, '言語': 3, 'コード': 4, 'ある': 19, '数': 16, 'C': 12, 'なる': 21, '行': 15, 'いる': 11, '扱う': 6, '書ける': 17})

    def test_dump_and_load(self):
        filepath = 'tests/tokenuid.dump'
        csv_list = ['tests/test2.csv']
        tuid = ld.TokenUID()
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
    tuid = ld.TokenUID()
    tuid.update(csv_list)
    ldata = ld.LearningData()

    def test_token_uid_list_2_vec_list(self):
        token_uids = [2, 0, 4, 10, 3]
        vecs = self.ldata.tuid_list_2_vec_list(
            token_uids, self.tuid.seq_no_uid)
        self.assertEqual(len(vecs[0]), 29)
        ans_vec0 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertTrue(np.allclose(vecs[0], ans_vec0))

    def test_calc_norm_tf_vector(self):
        token_uids = [2, 0, 2, 10, 3, 0, 2]
        vecs = self.ldata.tuid_list_2_vec_list(
            token_uids, self.tuid.seq_no_uid)
        tf = self.ldata.calc_norm_tf_vector(vecs)
        elem0 = 2.0 / len(token_uids)
        elem2 = 3.0 / len(token_uids)
        elem10 = 1.0 / len(token_uids)
        elem3 = 1.0 / len(token_uids)

        ans_vec0 = np.array(
            [elem0, 0, elem2, elem3, 0, 0, 0, 0, 0, 0, elem10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
        self.assertTrue(np.allclose(tf, ans_vec0))


if __name__ == '__main__':
    unittest.main()
