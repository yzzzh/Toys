import numpy as np
import tensorflow as tf
import scipy.misc

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
model_fn = 'tensorflow_inception_graph.pb'

with tf.gfile.FastGFile(model_fn,'rb') as f:
    graph_def  = tf.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tf.placeholder(np.float32,name='input')
image_mean = 117

t_preprocessed = tf.expand_dims(t_input - image_mean,0)
tf.import_graph_def(graph_def,{'input':t_preprocessed})


def savearray(img_array,img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s '%img_name)


def render_naive(t_obj, img0, iter_n, learning_rate=.1):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0.copy()
    for i in range(iter_n):
        grad, score = sess.run([t_grad, t_score], {t_input: img})
        grad /= grad.std() + 1e-8
        img += grad * learning_rate
        print('score(mean):%s' % score)
    savearray(img, 'naive.jpg')


name = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139
layer_output = graph.get_tensor_by_name('import/%s:0'%name)

img_noise = np.random.uniform(size=(224,224,3)) + 100.0
render_naive(layer_output[:,:,:,channel],img_noise,iter_n = 10)