import numpy as np
import tensorflow as tf

imgSize = 416
tiny = np.fromfile('yolov3.weights', np.float32)[5:]
anchors = np.array([10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]).reshape([9,2])

def conv2d(x, outfilters, ind, tiny, name_ind, name = 'conv', size = 3, stride = 1, is_training = False, trainable = False, batchnorm = True):
    def fixed_pad(x, stride=1):
        if stride>1:
            x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]]) #we only need to pad one pixel, the last row and column pad has no effect
        return x
    _,_,_, infilters = x.get_shape().as_list()
    if batchnorm == True:
        beta, gamma, mean, var = tiny[ind:ind+4*outfilters].reshape([4, outfilters])##
        ind = ind+4*outfilters
    else:
        b = tiny[ind:ind+outfilters]
        bias = tf.Variable(b, name = name + str(name_ind) + '/bias', trainable = trainable)
        ind = ind + outfilters
    num = size*size*infilters*outfilters
    w = np.transpose(tiny[ind:ind+num].reshape([outfilters, infilters, size, size]), (2,3,1,0))
    Weights = tf.Variable(w, name = name + str(name_ind) + '/kernel', trainable = trainable)
    ind = ind + num
    
    x = fixed_pad(x, stride)
    if batchnorm == True:
        xx = tf.nn.conv2d(x, Weights, [1,stride,stride,1], padding = 'SAME' if stride==1 else 'VALID')
        xx = tf.contrib.layers.batch_norm(xx, scale = True, 
                                          param_initializers={'beta':tf.constant_initializer(beta), 
                                                              'gamma':tf.constant_initializer(gamma), 
                                                              'moving_mean':tf.constant_initializer(mean), 
                                                              'moving_variance':tf.constant_initializer(var)}, 
                                                              is_training = is_training, 
                                                              trainable = trainable)
        return tf.nn.leaky_relu(xx, 0.1), ind, name_ind + 1
    else:
        return tf.nn.conv2d(x, Weights, [1, stride,stride,1], padding = 'SAME' if stride==1 else 'VALID') + bias, ind, name_ind + 1

def route(x1, x2):
    [_, H, W, _]= x1.get_shape().as_list()
    x1 = tf.image.resize_nearest_neighbor(x1, [H*2, W*2])
    return tf.concat([x1, x2], axis=3)

def darknet53_block(x, outfilters, ind, tiny, name_ind):
    shortcut = x
    x, ind, name_ind = conv2d(x, outfilters, ind, tiny, name_ind, size=1)
    x, ind, name_ind = conv2d(x, outfilters*2, ind, tiny, name_ind)
    return x + shortcut, ind, name_ind
def yolo_block(x, outfilters, ind, tiny, name_ind, name, num_class=80):
    x, ind, name_ind = conv2d(x, outfilters, ind, tiny, name_ind, name = name, size=1)
    x, ind, name_ind = conv2d(x, outfilters*2, ind, tiny, name_ind, name = name)
    x, ind, name_ind = conv2d(x, outfilters, ind, tiny, name_ind, name = name, size=1)
    x, ind, name_ind = conv2d(x, outfilters*2, ind, tiny, name_ind, name = name)
    x, ind, name_ind = conv2d(x, outfilters, ind, tiny, name_ind, name = name, size=1)
    route = x
    x, ind, name_ind = conv2d(x, outfilters*2, ind, tiny, name_ind, name = name)
    x, ind, name_ind = conv2d(x, 3*(5+num_class), ind, tiny, name_ind, name = name, size=1, batchnorm=False)
    return x, route, ind

ind = 0
x = tf.placeholder(tf.float32, [None,imgSize,imgSize,3])
out = []


'build darknet53'
with tf.variable_scope('body'):
    net, ind, name_ind = conv2d(x, 32, ind, tiny, name_ind = 0)
    net, ind, name_ind = conv2d(net, 64, ind, tiny, name_ind, stride=2)
    net, ind, name_ind = darknet53_block(net, 32, ind, tiny, name_ind)
    net, ind, name_ind = conv2d(net, 128, ind, tiny, name_ind, stride=2)
    
    for i in range(2):
        net, ind, name_ind = darknet53_block(net, 64, ind, tiny, name_ind)
    net, ind, name_ind = conv2d(net, 256, ind, tiny, name_ind, stride=2)
    for i in range(8):
        net, ind, name_ind = darknet53_block(net, 128, ind, tiny, name_ind)
        
    route1 = net
    net, ind, name_ind = conv2d(net, 512, ind, tiny, name_ind, stride=2)
    for i in range(8):
        net, ind, name_ind = darknet53_block(net, 256, ind, tiny, name_ind)
        
    route2 = net
    net, ind, name_ind = conv2d(net, 1024, ind, tiny, name_ind, stride=2)
    for i in range(4):
        net, ind, name_ind = darknet53_block(net, 512, ind, tiny, name_ind)

with tf.variable_scope('head'):
    #det1
    out1, route3, ind = yolo_block(net, 512, ind, tiny, name_ind = 0, name = 'det1_')
    out.append(out1)
    
    #det2
    net, ind, name_ind = conv2d(route3, 256, ind, tiny, name_ind = 0, name = 'det2_', size=1)
    net = route(net, route2)
    out2, route4, ind = yolo_block(net, 256, ind, tiny, name_ind, name = 'det2_')
    out.append(out2)
    
    #det3
    net, ind, name_ind = conv2d(route4, 128, ind, tiny, name_ind = 0, name = 'det3_', size=1)
    net = route(net, route1)
    out3, _, ind = yolo_block(net, 128, ind, tiny, name_ind, name = 'det3_')
    out.append(out3)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, './Weights/weightsInit.ckpt')