Deep Dream是Ｇoogle公司在2015年公布的一项有趣的技术。在训练好的卷积神经网络中，只需要设定几个参数，就可以通过这项技术生成一张图像。

假设输入网络的图像为x，网络输出的各个类别的概率为t，若一共有1000个分类，那么t是一个1000维的向量，代表了1000中类别的概率。假设香蕉类别的概率输出值为t[100]，则t[100]的值代表了香蕉的概率，t[100]的值越高，香蕉的概率就越高。那么我们反过来想，将t[100]作为我们的优化目标，不断调整图像的值，使得t[100]的值尽可能的大，同时，图像也越来越具有香蕉的特征。

总而言之，图片越像香蕉，那么t[100]的值就越大，那么t[100]的值越大，图片就越像香蕉，我们通过不断调整图像增大t[100]的值，从而得到香蕉的图像或者说具有香蕉的特征的图像。



```python
import numpy as np
import tensorflow as tf
import scipy.misc
import PIL.Image
from functools import partial

#这里我们使用已经训练好的inception模型
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
#导入模型
model_fn = 'tensorflow_inception_graph.pb'
with tf.gfile.FastGFile(model_fn,'rb') as f:
    graph_def  = tf.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tf.placeholder(np.float32,name='input')
#由于inception模型的输入是去平均化的,这里我们也同时减去117实现去平均化
image_mean = 117
#由于input的格式是[batch,height,width,depth],这里我们添加batch维度
t_preprocessed = tf.expand_dims(t_input - image_mean,0)
tf.import_graph_def(graph_def,{'input':t_preprocessed})

#对大张图片求梯度
#将大张的图片分成多个512*512的tile,分别对每个tile求梯度，最后将tile合并即可
def cal_grad_tiled(img,t_grad,tile_size=512):
    #对图像分别在x,y方向随机滑动,避免每个tile的边缘出现明显的分界线
    h,w = img.shape[:2]
    shift_x,shift_y = np.random.randint(tile_size,size=2)
    img_shift = np.roll(np.roll(img,shift_x,1),shift_y,0)
    grad = np.zeros_like(img)

    for y in range(0,max(h - tile_size//2,tile_size),tile_size):
        for x in range(0,max(w - tile_size//2,tile_size),tile_size):
            #在滑动后的img中截取一个tile,求出其梯度
            sub_img = img_shift[y:y+tile_size,x:x+tile_size]
            grad_sub = sess.run(t_grad,{t_input:sub_img})
            grad[y:y+tile_size,x:x+tile_size] = grad_sub
	#还原梯度图
    return np.roll(np.roll(grad,-shift_x,1),-shift_y,0)

#对图像进行放缩，这样做是为了保重放缩前后像素的范围不会改变
def resize(img,ratio):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float32(scipy.misc.imresize(img,ratio))
    img = img / 255 * (max - min) + min
    return img

#将array保存为图像
def savearray(img_array,img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s '%img_name)

#自定义一个5*5*3*3的卷积核
k = np.float32([1,4,6,4,1])
k = np.outer(k,k)
k5x5 = k[:,:,None,None] / k.sum() * np.eye(3,dtype=np.float32)

#拉普拉斯金字塔分解及融合
def lap_split(img):
    with tf.name_scope('split'):
        #对图像进行卷积及反卷积后，图像会变得模糊，即失去了高频成分
        #将原图像与处理后的图像相减即可得到高频成分
        #多次处理后即可得到高频成分的金字塔及失去了高频成分的图像
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

#将失去高频成分后的图像从上往下融合高频成分
def lap_merge(levels):
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img,k5x5*4,tf.shape(hi),[1,2,2,1],'SAME') + hi

    return img

#标准化，除以方差使得高低频成分分布的方差为1,减小了高低频成分的差值，使得高低频成分的分布更加均衡
def normalize_std(img,eps=1e-10):
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))

    return img / tf.maximum(std,eps)

#将图像分解为拉普拉斯金字塔,对每一层都进行标准化，最后将每一层融合，得到拉普拉斯算法处理后的图像
def lap_normalize(img,scale_n=4):
    img = tf.expand_dims(img,0)
    tlevles = lap_split_n(img,scale_n)
    tlevles = list(map(normalize_std,tlevles))
    out = lap_merge(tlevles)
    return out[0,:,:,:]

#只需要知道这是一个对Tensor定义的函数转化为对numpy.ndarray定义的函数即可
#即原函数输入和输出都是Tensor，转换后输入输入都是numpy.ndarray，且功能相同
def tffunc(*argtypes):
    placeholders = list(map(tf.placeholder, argtypes))

    def wrap(f):
        out = f(*placeholders)

        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

        return wrapper

    return wrap

def　　render_lapnorm(t_obj,img0,iter_n=10,learning_rate=1.,octave_n=3,octave_scale=1.4,lap_n=4):
    #t_obj是需要最大化的某个通道,我们的目标函数是这个通道的平均值,即最大化这个通道的平均值
    t_score = tf.reduce_mean(t_obj)
    #求目标函数对图像像素的梯度
    t_grad = tf.gradients(t_score,t_input)[0]
    #转换函数
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize,scale_n=lap_n))
	#尽量不对img0产生影响
    img = img0.copy()
    #总体过程:放大图像，求梯度，对梯度进行拉普拉斯函数处理，使得高低频成分分布均衡，最后更新图像
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
    #test.jpg自己找即可
    #将图像转化为numpy.ndarray的格式
    img0 = PIL.Image.open('test.jpg')
    img0 = np.float32(img0)
    # img0 = np.random.uniform(size=(224,224,3)) + 100.0
    #我们使用这个卷基层的第139个通道进行优化
    name = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139
    layer_output = graph.get_tensor_by_name('import/%s:0'%name)
    render_lapnorm(layer_output[:,:,:,channel],img0)
```

