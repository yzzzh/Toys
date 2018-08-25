import numpy as np
import tensorflow as tf
import scipy.misc
import PIL.Image
from functools import partial

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

k = np.float32([1,4,6,4,1])
k = np.outer(k,k)
k5x5 = k[:,:,None,None] / k.sum() * np.eye(3,dtype=np.float32)

def lap_split(img):
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img,k5x5,[1,2,2,1],'SAME')
        lo2 = tf.nn.conv2d_transpose(lo,k5x5*4,tf.shape(img),[1,2,2,1],'SAME')
        hi = img - lo2

    return lo,hi

def lap_split_n(img,n):
    levels = []
    for i in range(n):
        img,hi = lap_split(img)
        levels.append(hi)
    levels.append(img)

    return levels[::-1]

def lap_merge(levels):
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img,k5x5*4,tf.shape(hi),[1,2,2,1],'SAME') + hi

    return img

def normalize_std(img,eps=1e-10):
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))

    return img / tf.maximum(std,eps)

def lap_normalize(img,scale_n=4):
    img = tf.expand_dims(img,0)
    tlevles = lap_split_n(img,scale_n)
    tlevles = list(map(normalize_std,tlevles))
    out = lap_merge(tlevles)
    return out[0,:,:,:]

def tffunc(*argtypes):
    placeholders = list(map(tf.placeholder, argtypes))

    def wrap(f):
        out = f(*placeholders)

        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

        return wrapper

    return wrap

def render_lapnorm(t_obj,img0,iter_n=10,learning_rate=1.5,octave_n=3,octave_scale=1.4,lap_n=4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score,t_input)[0]
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize,scale_n=lap_n))

    img = img0.copy()
    for octave in range(octave_n):
        if octave > 0:
            img = resize(img,octave_scale)
        for i in range(iter_n):
            g = cal_grad_tiled(img,t_grad)
            g = lap_norm_func(g)
            img += g * learning_rate
            print('round:%s,iter:%s'%(octave,i))

    savearray(img,'lapnorm.jpg')


if __name__ == '__main__':
    img0 = PIL.Image.open('test.jpg')
    img0 = np.float32(img0)
    # img1 = np.random.uniform(size=(224,224,3)) + 100.0
    name = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    layer_output = graph.get_tensor_by_name('import/%s:0'%name)
    render_lapnorm(layer_output[:,:,:,channel],img0)