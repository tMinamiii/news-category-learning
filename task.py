import numpy as np

import constant_values as c
import double_layer_nn as dlnn
import vectorize.learning_data as ld


def main():
    tuid = ld.load(c.TUID_FILE)
    td = ld.load(c.DATA_FILE)
    np.random.shuffle(td)
    tdlen = len(td)
    boundary = int(tdlen * c.TRAINING_DATA_RATIO)
    pd = td[boundary:tdlen - 1]
    td = td[0:boundary - 1]
    print(len(pd))
    print(len(td))
    num_categories = len(tuid.categories)
    predict_label = pd[:, 0].tolist()
    predict_data = pd[:, 1].tolist()

    nn = dlnn.DoubleLayerNetwork(c.LEARNING_RATIO, c.NUM_UNITS,
                                 c.SVD_DIMENSION, num_categories, c.LOG_FILE)
    i = 0
    for _ in range(c.TOTAL_STEP):
        i += 1
        np.random.shuffle(td)
        batch_label = td[:c.BATCH_SIZE, 0].tolist()
        batch_data = td[:c.BATCH_SIZE, 1].tolist()

        nn.sess.run(nn.train_step, feed_dict={
            nn.x: batch_data, nn.t: batch_label, nn.keep_prob: 0.5})
        if i % 100 == 0:
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss, nn.accuracy],
                feed_dict={nn.x: predict_data,
                           nn.t: predict_label,
                           nn.keep_prob: 1.0})
            print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
            nn.writer.add_summary(summary, i)


if __name__ == '__main__':
    main()
