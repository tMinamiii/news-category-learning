import tensorflow as tf
import learning_data as ld

#tuid = ld.TokenUID()
# ld.load('tuid/2017-10-05.tuid')
train_ld = ld.load('ldata/2017-10-06.ldata')
tuid = train_ld.token_uid
vec_dim = tuid.seq_no_uid + 1
num_units = 1024
num_categories = len(tuid.categories)
train_label = train_ld.train_label
train_data = train_ld.train_vec

x = tf.placeholder(tf.float32, [None, vec_dim])

w1 = tf.Variable(tf.truncated_normal([vec_dim, num_units]))
b1 = tf.Variable(tf.zeros(num_units))
hidden = tf.nn.relu(tf.matmul(x, w1) + b1)

w0 = tf.Variable(tf.zeros([num_units, num_categories]))
b0 = tf.Variable(tf.zeros([num_categories]))
p = tf.nn.sigmoid(tf.matmul(hidden, w0) + b0)

t = tf.placeholder(tf.float32, [None, num_categories])
loss = -1 * tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdadeltaOptimizer().minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# batch_size = 100
# batch_count = 0
# batch_data = train_data[batch_count:batch_count + batch_size]
# batch_count += batch_size
i = 0
for _ in range(5000):
    i += 1
    sess.run(train_step, feed_dict={x: train_data, t: train_label})
    if i % 100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={
                                     x: train_data, t: train_label})
        print('Step: %d, Loss: %f, Accuracy: %f' % (1, loss_val, acc_val))