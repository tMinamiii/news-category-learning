import numpy as np
from learning import double_layer_nn as dlnn
import util


def main():
    meta = util.pickle_load(util.METADATA_FILE)
    label_and_data = util.pickle_load(util.DATA_FILE)
    np.random.shuffle(label_and_data)
    length = len(label_and_data)
    boundary = int(length * util.TRAINING_DATA_RATIO)
    test = label_and_data[boundary:]
    train = label_and_data[0:boundary - 1]
    print(len(test))
    print(len(train))
    num_categories = len(meta.categories)
    predict_label = test[:, 0].tolist()
    predict_data = test[:, 1].tolist()

    nn = dlnn.DoubleLayerNetwork(util.LEARNING_RATE, util.NUM_UNITS,
                                 util.PCA_DIMENSION, num_categories,
                                 util.LOG_FILE)
    i = 0
    for _ in range(util.TOTAL_STEP):
        i += 1
        np.random.shuffle(train)
        batch_label = train[:util.BATCH_SIZE, 0].tolist()
        batch_data = train[:util.BATCH_SIZE, 1].tolist()

        nn.sess.run(nn.train_step, feed_dict={
            nn.x: batch_data, nn.t: batch_label, nn.keep_prob: util.KEEP_PROB})
        if i % 100 == 0:
            summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss, nn.accuracy],
                feed_dict={nn.x: predict_data,
                           nn.t: predict_label,
                           nn.keep_prob: 1.0})
            print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
            nn.writer.add_summary(summary, i)