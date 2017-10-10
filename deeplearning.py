import tensorflow as tf
import learning_data as ld
import numpy as np

tuid = ld.load('ldata/2017-10-10.tuid')
td = ld.load('ldata/2017-10-10.svdtd')
np.random.shuffle(td)
tdlen = len(td)
boundary = int(tdlen / 10 * 8)
pd = td[boundary:tdlen - 1]
td = td[0:boundary - 1]
print(len(pd))
print(len(td))
# vec_dim = tuid.seq_no_uid + 1
vec_dim = len(td[:, 1][0])
num_units = 8192
num_categories = len(tuid.categories)
predict_label = pd[:, 0].tolist()
predict_data = pd[:, 1].tolist()

x = tf.placeholder(tf.float32, [None, vec_dim])

w1 = tf.Variable(tf.truncated_normal([vec_dim, num_units]))
b1 = tf.Variable(tf.zeros(num_units))
hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.truncated_normal([num_units, vec_dim]))
b2 = tf.Variable(tf.zeros(vec_dim))
hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)
'''
keep_prob = tf.placeholder(tf.float32)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

w3 = tf.Variable(tf.truncated_normal([vec_dim, num_units]))
b3 = tf.Variable(tf.zeros(num_units))
hidden3 = tf.nn.relu(tf.matmul(hidden2, w3) + b3)

w4 = tf.Variable(tf.truncated_normal([num_units, vec_dim]))
b4 = tf.Variable(tf.zeros(vec_dim))
hidden4 = tf.nn.relu(tf.matmul(hidden3, w4) + b4)
'''
w0 = tf.Variable(tf.zeros([vec_dim, num_categories]))
b0 = tf.Variable(tf.zeros([num_categories]))
p = tf.nn.softmax(tf.matmul(hidden2, w0) + b0)

t = tf.placeholder(tf.float32, [None, num_categories])
loss = -1 * tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdadeltaOptimizer(0.01).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 200
i = 0
for _ in range(20000):
    i += 1
    np.random.shuffle(td)
    batch_label = td[:batch_size, 0].tolist()
    batch_data = td[:batch_size, 1].tolist()

    sess.run(train_step, feed_dict={
             x: batch_data, t: batch_label})
    # sess.run(train_step, feed_dict={x: train_data, t: train_label})
    if i % 10 == 0:
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={
                                     x: predict_data, t: predict_label})
        print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
