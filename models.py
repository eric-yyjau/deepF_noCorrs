"""Contains network architectures"""
import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
from UCN.networks import UniversalCorrepondenceNetwork

class HomographyNet(object):

    def __init__(self, dim=64, ksize=3, use_bn=True, use_dropout=True,
                 out_dim=9, weight_decay=0.01, use_idx=True, use_coor=False,
                 norm_method='norm', use_reconstruction_module=True):
        self.name = "homography_net"
        self.dim = dim
        self.ksize = ksize
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.weight_decay = weight_decay
        self.use_idx = use_idx
        self.use_coor = use_coor
        self.norm = norm_method
        self.use_reconstruction_module = use_reconstruction_module
        print("HomographNet Use coord:%s"%self.use_coor)
        if self.use_reconstruction_module:
            self.out_dim = 8
        else:
            self.out_dim = out_dim

    def normalize_output(self, x):
        if self.norm == 'norm':
            print("[model]Using L2 norm to normalize the output")
            return x / (tf.norm(x, axis=1, keep_dims=True) + 1e-8)
        elif self.norm == 'abs':
            print("[model]Using maximum absolute value to normalize the output")
            return x / (tf.reduce_max(tf.abs(x), axis=1, keep_dims=True) + 1e-8)
        elif self.norm == 'last':
            print("[model]Using the last index to normalize the output")
            return x / (tf.expand_dims(tf.reshape(x[:,-1],[-1]), axis=1) + 1e-8)
        else:
            raise Exception("Unrecognized normaliztion method:%s"%self.norm)

    def conv2d(self, x, dim, ksizes, strides, padding, activation):
        if self.weight_decay > 0:
            return tf.layers.conv2d(
                    x, dim, ksizes, strides,
                    padding=padding, activation=activation,
                    kernel_regularizer=tc.layers.l2_regularizer(scale=self.weight_decay))
        else:
            return tf.layers.conv2d(
                    x, dim, ksizes, strides,
                    padding=padding, activation=activation)

    def fetch_idx(self, orig_idx, new_idx):
        c = new_idx.get_shape()[-1]
        new_idx = tf.cast(tf.divide(new_idx, c), tf.int64)
        out = tf.gather(params=tf.reshape(orig_idx, shape=[-1]), indices=new_idx)
        print("Orig:%s\tNew:%s\tOut:%s"\
              %(orig_idx.get_shape(), new_idx.get_shape(), out.get_shape()))
        return out

    def reconstruction_module(self, x):
        print("Use structural output layer")
        def get_rotation(rx, ry, rz):
            # normalize input?
            R_x = tf.stack([
                [1.,    0.,             0.],
                [0.,    tf.cos(rx),    -tf.sin(rx)],
                [0.,    tf.sin(rx),     tf.cos(rx)]
            ])
            R_y = tf.stack([
                [tf.cos(ry),    0.,    -tf.sin(ry)],
                [0.,            1.,     0.],
                [tf.sin(ry),    0.,     tf.cos(ry)]
            ])
            R_z = tf.stack([
                [tf.cos(rz),    -tf.sin(rz),    0.],
                [tf.sin(rz),    tf.cos(rz),     0.],
                [0.,            0.,             1.]
            ])
            R = tf.matmul(R_x, tf.matmul(R_y, R_z))
            return R

        def get_inv_intrinsic(f):
            return tf.stack([
                [-1/(f+1e-8),   0.,             0.],
                [0.,            -1/(f+1e-8),    0.],
                [0.,            0.,             1.]
            ])

        def get_translate(tx, ty, tz):
            return tf.stack([
                [0.,  -tz, ty],
                [tz,  0,   -tx],
                [-ty, tx,  0]
            ])

        def get_linear_comb(f0, f1, f2, f3, f4, f5, cf1, cf2):
            return tf.stack([
                [f0,            f1,            f2],
                [f3,            f4,            f5],
                [cf1*f0+cf2*f3, cf1*f1+cf2*f4, cf1*f2+cf2*f5]
            ])

        def get_fmat(x):
            # Note: only need out-dim = 8
            K1_inv = get_inv_intrinsic(x[0])
            K2_inv = get_inv_intrinsic(x[1])
            R  = get_rotation(x[2], x[3], x[4])
            T  = get_translate(x[5], x[6], x[7])
            F  = tf.matmul(K2_inv,
                    tf.matmul(R, tf.matmul(T, K1_inv)))
            flat = tf.reshape(F, [-1])

            # to get the last row as linear combination of first two rows
            # new_F = get_linear_comb(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
            # new_F = get_linear_comb(flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], x[6], x[7])
            # flat = tf.reshape(new_F, [-1])
            print ("Using reconstruction layer")
            return flat

        print("Using structural F-matrix output")
        out = tf.map_fn(get_fmat, x)

        return out


    def __call__(self, x1, x2, img_shape, is_training, reuse=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            print(x1.get_shape())
            print(tf.size(x1))
            
            '''
            # UCN model, uncomment this part for UCN model
            ucn = UniversalCorrepondenceNetwork(x1, x2, img_shape)
            feature1, feature2 = ucn(x1,x2, img_shape)
            x = tf.concat([feature1, feature2], axis=3)
            print ('feature vector: ', x.shape)
            '''
            # single model
            x = tf.concat([x1, x2], axis=3)
            
            # uncomment this portion to use the single stream regressor network
            def get_grid(_):
                ret = tf.range(x.get_shape()[1] * x.get_shape()[2])
                return ret
            x_idx = tf.map_fn(get_grid, tf.range(tf.shape(x)[0]))
            print(x_idx.get_shape())
            # x_idx = tf.range(tf.size(x)/x.get_shape()[-1])
        
            # Group 1 (128x128)
            conv1_1 = self.conv2d(x, self.dim, [self.ksize, self.ksize], [1, 1],
                                  padding='SAME', activation=None)
            if self.use_bn:
                conv1_1 = tf.layers.batch_normalization(conv1_1, training=is_training)
            conv1_1 = tf.nn.relu(conv1_1)

            conv1_2 = self.conv2d(conv1_1, self.dim, [self.ksize, self.ksize], [1, 1],
                                  padding='SAME', activation=None)
            if self.use_bn:
                conv1_2 = tf.layers.batch_normalization(conv1_2, training=is_training)
            conv1_2 = tf.nn.relu(conv1_2)

            conv1, conv1_idx = tf.nn.max_pool_with_argmax(
                    input=conv1_2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
            conv1_idx = self.fetch_idx(x_idx, conv1_idx)
            # print(conv1_idx.get_shape())
            # print(conv1.get_shape())

            # Group 2 (64x64)
            conv2_1 = self.conv2d(conv1, self.dim, [self.ksize, self.ksize], [1, 1],
                                  padding='SAME', activation=None)
            if self.use_bn:
                conv2_1 = tf.layers.batch_normalization(conv2_1, training=is_training)
            conv2_1 = tf.nn.relu(conv2_1)
            conv2_2 = self.conv2d(conv2_1, self.dim, [self.ksize, self.ksize], [1, 1],
                                  padding='SAME', activation=None)
            if self.use_bn:
                conv2_2 = tf.layers.batch_normalization(conv2_2, training=is_training)
            conv2_2 = tf.nn.relu(conv2_2)
            
            conv2, conv2_idx = tf.nn.max_pool_with_argmax(
                    input=conv2_2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
            conv2_idx = self.fetch_idx(conv1_idx, conv2_idx)
            print(conv2_idx.get_shape())
            print(conv2.get_shape())
            
            # Group 3 (32x32)
            conv3_1 = self.conv2d(conv2, self.dim*2, [self.ksize, self.ksize], [1, 1],
                                  padding='SAME', activation=None)
            if self.use_bn:
                conv3_1 = tf.layers.batch_normalization(conv3_1, training=is_training)
            conv3_1 = tf.nn.relu(conv3_1)
            conv3_2 = self.conv2d(conv3_1, self.dim*2, [self.ksize, self.ksize], [1, 1],
                                  padding='SAME', activation=None)
            if self.use_bn:
                conv3_2 = tf.layers.batch_normalization(conv3_2, training=is_training)
            conv3_2 = tf.nn.relu(conv3_2)
            '''
            conv3, conv3_idx = tf.nn.max_pool_with_argmax(
                input=conv3_2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
            conv3_idx = self.fetch_idx(conv2_idx, conv3_idx)
            print(conv3_idx.get_shape())
            print(conv3.get_shape())
            '''
            # Group 4 (16x16)
            conv4_1 = self.conv2d(conv3_2, self.dim*2, [self.ksize, self.ksize], [1, 1],
                                  padding='SAME', activation=None)
            if self.use_bn:
                conv4_1 = tf.layers.batch_normalization(conv4_1, training=is_training)
            conv4_1 = tf.nn.relu(conv4_1)
            conv4_2 = self.conv2d(conv4_1, self.dim*2, [self.ksize, self.ksize], [1, 1],
                                  padding='SAME', activation=None)
            if self.use_bn:
                conv4_2 = tf.layers.batch_normalization(conv4_2, training=is_training)
            conv4_2 = tf.nn.relu(conv4_2)
            '''
            conv4, conv4_idx = tf.nn.max_pool_with_argmax(
                    input=conv4_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            conv4_idx = self.fetch_idx(conv3_idx, conv4_idx)
            print(conv4_idx.get_shape())
            print(conv4.get_shape())
            '''
            # Group 5
            conv5_1 = self.conv2d(conv4_2, self.dim*2, [self.ksize, self.ksize], [1, 1],
                                  padding='SAME', activation=None)
            if self.use_bn:
                conv5_1 = tf.layers.batch_normalization(conv5_1, training=is_training)
            conv5_1 = tf.nn.relu(conv5_1)
            conv5_2 = self.conv2d(conv5_1, self.dim*2, [self.ksize, self.ksize], [1, 1],
                                  padding='SAME', activation=None)
            if self.use_bn:
                conv5_2 = tf.layers.batch_normalization(conv5_2, training=is_training)
            conv5_2 = tf.nn.relu(conv5_2)
            '''
            conv5, conv5_idx = tf.nn.max_pool_with_argmax(
                    input=conv5_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            conv5_idx = self.fetch_idx(conv4_idx, conv5_idx)
            print(conv5_idx.get_shape())
            print(conv5.get_shape())
            '''
            conv5, conv5_idx = tf.nn.max_pool_with_argmax(
                    input=conv5_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            conv5_idx = self.fetch_idx(conv1_idx, conv5_idx)
            print(conv5_idx.get_shape())
            print(conv5.get_shape())
            
            if self.use_coor:
                conv5_x  = tf.cast(conv5_idx / x.get_shape()[1], tf.float32)
                conv5_y  = tf.cast(conv5_idx % x.get_shape()[1], tf.float32)
                conv5_x = conv5_x / tf.cast(tf.shape(x)[1], tf.float32)
                conv5_y = conv5_y / tf.cast(tf.shape(x)[2], tf.float32)
                # TODO: normalize the indices
                conv5 = tf.concat([conv5, conv5_x, conv5_y], axis=3)
                print("Use corrdinate:(x,y)")
                print(conv5.get_shape())
            elif self.use_idx:
                # TODO: normalize the indices
                conv5_idx = tf.cast(conv5_idx, tf.float32)
                conv5_idx = conv5_idx / tf.cast(tf.shape(x)[1] * tf.shape(x)[2], tf.float32)
                conv5 = tf.concat([conv5, conv5_idx], axis=3)
                print("Use idx (0,1) normalized.")
                print(conv5.get_shape())
            
            # Flatten and make decision
            flat = tcl.flatten(conv5)
            print(flat.get_shape())
            dense1 = tf.layers.dense(flat, 1024, activation=tf.nn.relu)
            if self.use_dropout:
                dense1 = tf.layers.dropout(dense1, rate=0.5)
            out = tf.layers.dense(dense1, self.out_dim)

            if self.use_reconstruction_module:
                out = self.reconstruction_module(out)

            out = self.normalize_output(out)

        return out

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


