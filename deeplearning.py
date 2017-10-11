import tensorflow as tf
import learning_data as ld
import numpy as np

NUM_UNITS = 8192


class DoubleLayerNetwork:
    def __init__(self, num_units, vec_dim, num_categories):
        with tf.Graph().as_default():
            self.prepare_model(num_units, vec_dim, num_categories)
            self.prepare_session()

    def prepare_model(self, num_units, vec_dim, num_categories):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, vec_dim])

        with tf.name_scope('hidden1'):
            w1 = tf.Variable(tf.truncated_normal([vec_dim, num_units]))
            b1 = tf.Variable(tf.zeros(num_units))
            hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)

        with tf.name_scope('hidden2'):
            w2 = tf.Variable(tf.truncated_normal([num_units, vec_dim]))
            b2 = tf.Variable(tf.zeros(vec_dim))
            hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)

        with tf.name_scope('output'):
            w0 = tf.Variable(tf.zeros([vec_dim, num_categories]))
            b0 = tf.Variable(tf.zeros([num_categories]))
            p = tf.nn.softmax(tf.matmul(hidden2, w0) + b0)

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, num_categories])
            loss = -1 * tf.reduce_sum(t * tf.log(p))
            train_step = tf.train.AdadeltaOptimizer(0.005).minimize(loss)

        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.histogram('weights_hidden2', w2)
        tf.summary.histogram('biases_hidden2', b2)
        tf.summary.histogram('weights_hidden1', w1)
        tf.summary.histogram('biases_hidden1', b1)
        tf.summary.histogram('weights_output', w0)
        tf.summary.histogram('biases_output', b0)

        self.x = x
        self.t = t
        self.p = p
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('/tmp/yn_categories_logs', sess.graph)

        self.sess = sess
        self.summary = summary
        self.writer = writer


def main():
    tuid = ld.load('ldata/2017-10-11.tuid')
    td = ld.load('ldata/2017-10-11.svdtd')
    np.random.shuffle(td)
    tdlen = len(td)
    boundary = int(tdlen / 10 * 8)
    pd = td[boundary:tdlen - 1]
    td = td[0:boundary - 1]
    print(len(pd))
    print(len(td))
    vec_dim = len(td[:, 1][0])
    num_categories = len(tuid.categories)
    predict_label = pd[:, 0].tolist()
    predict_data = pd[:, 1].tolist()

    nn = DoubleLayerNetwork(NUM_UNITS, vec_dim, num_categories)
    batch_size = 200
    i = 0
    for _ in range(50000):
        i += 1
        np.random.shuffle(td)
        batch_label = td[:batch_size, 0].tolist()
        batch_data = td[:batch_size, 1].tolist()

        nn.sess.run(nn.train_step, feed_dict={
            nn.x: batch_data, nn.t: batch_label})
        if i % 10 == 0:
            loss_val, acc_val = nn.sess.run([nn.loss, nn.accuracy], feed_dict={
                nn.x: predict_data, nn.t: predict_label})
            print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))


if __name__ == '__main__':
    main()
