import numpy as np
import tensorflow as tf

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

layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]

for layer in layers:
    print(layer)

name = 'mixed4d_3x3_bottleneck_pre_relu'
print('shape of %s : %s'%(name,str(graph.get_tensor_by_name('import/'+name+':0').get_shape())))

