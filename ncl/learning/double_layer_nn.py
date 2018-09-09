import tensorflow as tf


class DoubleLayerNetwork:
    def __init__(self, learning_rate, num_units1,
                 num_units2, num_units3, num_units4,
                 vec_dim, num_categories, logfile):
        with tf.Graph().as_default():
            self.prepare_model(learning_rate, num_units1, num_units2,
                               num_units3, num_units4,
                               vec_dim, num_categories)
            self.prepare_session(logfile)

    def prepare_model(self, learning_rate,
                      num_units1, num_units2,
                      num_units3, num_units4, vec_dim, num_categories):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, vec_dim])

        with tf.name_scope('hidden1'):
            w1 = tf.Variable(tf.truncated_normal([vec_dim, num_units1]))
            b1 = tf.Variable(tf.zeros(num_units1))
            hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)

        # with tf.name_scope('hidden2'):
        #     w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
        #     b2 = tf.Variable(tf.zeros(num_units2))
        #     hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)

        # with tf.name_scope('hidden3'):
        #     w3 = tf.Variable(tf.truncated_normal([num_units2, num_units3]))
        #     b3 = tf.Variable(tf.zeros(num_units3))
        #     hidden3 = tf.nn.relu(tf.matmul(hidden2, w3) + b3)

        # with tf.name_scope('hidden4'):
        #     w4 = tf.Variable(tf.truncated_normal([num_units3, num_units4]))
        #     b4 = tf.Variable(tf.zeros(num_units4))
        #     hidden4 = tf.nn.relu(tf.matmul(hidden3, w4) + b4)

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

        with tf.name_scope('output'):
            w0 = tf.Variable(tf.zeros([num_units1, num_categories]))
            b0 = tf.Variable(tf.zeros([num_categories]))
            p = tf.nn.softmax(tf.matmul(hidden1_drop, w0) + b0)

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, num_categories])
            loss = -1 * tf.reduce_sum(t * tf.log(p))
            train_step = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(loss)

        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        # tf.summary.histogram('weights_hidden2', w2)
        # tf.summary.histogram('biases_hidden2', b2)
        # tf.summary.histogram('weights_hidden1', w1)
        # tf.summary.histogram('biases_hidden1', b1)
        # tf.summary.histogram('weights_output', w0)
        # tf.summary.histogram('biases_output', b0)

        self.x = x
        self.t = t
        self.p = p
        self.keep_prob = keep_prob
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy

    def prepare_session(self, logfile):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logfile, sess.graph)

        self.sess = sess
        self.summary = summary
        self.writer = writer
