import numpy as np
import pickle
from learnign import double_layer_nn as dlnn
import utils as u


def pickle_load(filepath) -> None:
    with open(filepath, mode='rb') as f:
        data = pickle.load(f)
    return data


def main():
    meta = pickle_load(u.METADATA_FILE)
    label_and_data = pickle_load(u.DATA_FILE)
    np.random.shuffle(label_and_data)
    length = len(label_and_data)
    boundary = int(length * u.TRAINING_DATA_RATIO)
    test = label_and_data[boundary:]
    train = label_and_data[0:boundary - 1]
    print(len(test))
    print(len(train))
    num_categories = len(meta.categories)
    predict_label = test[:, 0].tolist()
    predict_data = test[:, 1].tolist()

    nn = dlnn.DoubleLayerNetwork(u.LEARNING_RATE, u.NUM_UNITS,
                                 u.PCA_DIMENSION, num_categories, u.LOG_FILE)
    i = 0
    for _ in range(u.TOTAL_STEP):
        i += 1
        np.random.shuffle(train)
        batch_label = train[:u.BATCH_SIZE, 0].tolist()
        batch_data = train[:u.BATCH_SIZE, 1].tolist()

        nn.sess.run(nn.train_step, feed_dict={
            nn.x: batch_data, nn.t: batch_label, nn.keep_prob: u.KEEP_PROB})
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
