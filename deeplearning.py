import tensorflow as tf
from learning_data import TokenUID
from learning_data import LearningData

tuid = TokenUID()
tuid.load('tuid/2017-10-05.tuid')
train_ld = LearningData(tuid)
train_label, train_data, predict_label, predict_data = train_ld.make()

vec_dim = tuid.seq_no_uid
num_units = 1024
num_categories = len(tuid.categories)
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
sess.run(train_step, feed_dict={x: train_data, t: train_label})
loss_val, acc_val = sess.run([loss, accuracy], feed_dict={
                             x: train_data, t: train_label})
print('Step: %d, Loss: %f, Accuracy: %f' % (1, loss_val, acc_val))
