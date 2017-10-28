import numpy as np

import constant_values as c
import double_layer_nn as dlnn
import vectorize.vectorizer as ld


def main():
    prep = ld.load(c.PREPROCESSED_FILE)
    label_and_data = ld.load(c.DATA_FILE)
    np.random.shuffle(label_and_data)
    length = len(label_and_data)
    boundary = int(length * c.TRAINING_DATA_RATIO)
    test = label_and_data[boundary:]
    train = label_and_data[0:boundary - 1]
    print(len(test))
    print(len(train))
    num_categories = len(prep.categories)
    predict_label = test[:, 0].tolist()
    predict_data = test[:, 1].tolist()

    nn = dlnn.DoubleLayerNetwork(c.LEARNING_RATE, c.NUM_UNITS,
                                 c.PCA_DIMENSION, num_categories, c.LOG_FILE)
    i = 0
    for _ in range(c.TOTAL_STEP):
        i += 1
        np.random.shuffle(train)
        batch_label = train[:c.BATCH_SIZE, 0].tolist()
        batch_data = train[:c.BATCH_SIZE, 1].tolist()

        nn.sess.run(nn.train_step, feed_dict={
            nn.x: batch_data, nn.t: batch_label, nn.keep_prob: c.KEEP_PROB})
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
