import numpy as np
import tensorflow as tf
import scipy.misc

graph = tf.Graph()
sess = tf.Session(graph=graph)
model_fn = 'tensorflow_inception_graph.pb'

with tf.gfile.FastGFile(model_fn,'rb') as f:
    graph_def  = tf.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tf.placeholder(np.float32,name='input')
image_mean = 117

t_preprocessed = tf.expand_dims(t_input - image_mean,0)
tf.import_graph_def(graph_def,{'input':t_preprocessed})
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

def cal_grad_tiled(img,t_grad,tile_size=512):
    h,w = img.shape[:2]
    shift_x,shift_y = np.random.randint(tile_size,size=2)
    img_shift = np.roll(np.roll(img,shift_x,1),shift_y,0)
    grad = np.zeros_like(img)

    for y in range(0,max(h - tile_size//2,tile_size),tile_size):
        for x in range(0,max(w - tile_size//2,tile_size),tile_size):
            sub_img = img_shift[y:y+tile_size,x:x+tile_size]
            grad_sub = sess.run(t_grad,{t_input:sub_img})
            grad[y:y+tile_size,x:x+tile_size] = grad_sub

    return np.roll(np.roll(grad,-shift_x,1),-shift_y,0)

def resize(img,ratio):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img,ratio))
    img = img / 255 * (max - min) + min

    return img


def savearray(img_array,img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s '%img_name)


def render_multiscale(t_obj,img0,iter_n=20,learning_rate=1.0,octave_n=3,octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score,t_input)[0]

    img = img0.copy()
    for octave in range(octave_n):
        img = resize(img,octave_scale)

        for i in range(iter_n):
            g = cal_grad_tiled(img,t_grad)
            g /= g.std() + 1e-8
            img += g * learning_rate
            print('.')

    savearray(img,'multiscale.jpg')

if __name__ == '__main__':
    name = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    img_noise = np.random.uniform(size=(224,224,3)) + 100.0
    layer_output = graph.get_tensor_by_name('import/%s:0'%name)
    render_multiscale(layer_output[:,:,:,channel],img_noise)