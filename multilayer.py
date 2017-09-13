import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

fig = plt.figure()

N = 300
X, y = datasets.make_moons(N, noise=0.3)

for i in range(150):
    if y[i] == 1:
        plt.plot(X[i][0], X[i][1], 'ro')
    elif y[i] == 0:
        plt.plot(X[i][0], X[i][1], 'bo')


Y = y.reshape(N, 1)
X_train, X_test, Y_train, Y_test, =  train_test_split(X, Y, train_size=0.8, random_state=0)

#tf.truncated_normal 切断正規分布から乱数を出力.重みは学習によって最適な値に出力されるので,最初の値は乱数で十分
num_hidden_0 = 4
x = tf.placeholder(tf.float32, shape=[None, 2])
W_0 = tf.Variable(tf.truncated_normal([2, num_hidden_0]))
b_0 = tf.Variable(tf.zeros([num_hidden_0]))
h_0 = tf.nn.sigmoid(tf.matmul(x, W_0) + b_0)

num_hidden_1 = 6
W_1 = tf.Variable(tf.truncated_normal([num_hidden_0, num_hidden_1]))
b_1 = tf.Variable(tf.zeros([num_hidden_1]))
h_1 = tf.nn.sigmoid(tf.matmul(h_0, W_1) + b_1)

V = tf.Variable(tf.truncated_normal([num_hidden_1, 1]))
c = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(tf.matmul(h_1, V) + c)

t = tf.placeholder(tf.float32, shape=[None, 1])
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1-t) * tf.log(1-y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 30
n_batches = N

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(500):
    X_, Y_ = shuffle(X_train, Y_train)
   
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        #[start:end]は、startオフセットからend-1オフセットまでのシーケンスを抽出する
        sess.run(train_step, feed_dict={x: X_[start:end], t: Y_[start:end]})

accuracy_rate = accuracy.eval(session=sess, feed_dict={x: X_test, t: Y_test})
print('accuracy:', accuracy_rate)

W0_val, W1_val, V_val = sess.run([W_0, W_1, V]) 
b0_val, b1_val, c_val = sess.run([b_0, b_1, c])

def classifying_input(X):
    return np.dot(X, V_val) + c_val

def sigmoid(a):
    h = 1 / ( 1 + np.exp(-a))
    return h

def predict(X):
    h_0 = np.tanh(np.dot(X, W0_val) + b0_val)
    h_2 = np.tanh(np.dot(h_0, W1_val) + b1_val)

    return np.where(1 / (1 + np.exp(-classifying_input(h_2))) >= 0.5, 1, 0)

def classifying_plot(X, y, resolution=0.02):
    
    markers = ('s', 'x')
    colors = ('red', 'blue')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),\
            np.arange(x2_min, x2_max, resolution))

    Z = predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

classifying_plot(X_test, Y_test)

plt.xlim([-2.0, 3.0])
plt.ylim([-1.5, 2.0])
plt.show()
