import tensorflow as tf
import time

class Generator:
    def __init__(self, depths=[128, 512, 1024, 2048, 1024, 512, 128], s_size=4):
        self.depths = [1] + depths + [1]
        self.s_size = s_size
        self.reuse = False

    def __call__(self, inputs, batch_size,  training=False):
        self.batch_size = batch_size

        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('g', reuse=self.reuse):

            with tf.variable_scope('Conv1'):
                g_filter_conv1 = tf.get_variable('weights', [1, 15, self.depths[0], self.depths[1]], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
                outputs1 = tf.nn.conv2d(inputs, g_filter_conv1, strides=[1, 1, 1, 1], padding='SAME')
                outputs1 = tf.nn.relu(outputs1, name='outputs')

            with tf.variable_scope('Conv2'):
                g_filter_conv2 = tf.get_variable('weights', [1, 5, self.depths[1], self.depths[2]], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
                outputs2 = tf.nn.conv2d(outputs1, g_filter_conv2, strides=[1, 1, 1, 1], padding='SAME')
                outputs2 = tf.nn.relu(outputs2, name='outputs')

            with tf.variable_scope('deConv1-clear'):
                clears1 = tf.layers.conv2d_transpose(outputs2, self.depths[7], kernel_size=[1,5], strides=(1,1), padding='SAME')
                clears1 = tf.nn.relu(clears1, name='clears')

            with tf.variable_scope('deConv2-clear'):
                clears2 = tf.layers.conv2d_transpose(clears1, self.depths[8], kernel_size=[1,15], strides=(1,1), padding='SAME')

            with tf.variable_scope('deConv1-noise'):
                noises1 = tf.layers.conv2d_transpose(outputs2, self.depths[7], kernel_size=[1,5], strides=(1,1), padding='SAME')
                noises1 = tf.nn.relu(noises1, name='noises')

            with tf.variable_scope('deConv2-noise'):
                noises2 = tf.layers.conv2d_transpose(noises1, self.depths[8], kernel_size=[1,15], strides=(1,1), padding='SAME')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        clears_outputs = tf.reshape(clears2, [self.batch_size, 1, 5000, 1])
        noises_outputs = tf.reshape(noises2, [self.batch_size, 1, 5000, 1])
        return [clears_outputs, noises_outputs]

class Discriminator:
    def __init__(self, depths=[64, 128, 256, 512]):
        self.depths = [1] + depths
        self.reuse = False

    def __call__(self, inputs, batch_size, training=False,  name=''):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        inputs = tf.convert_to_tensor(inputs)
        
        with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
            # convolution x 4
            with tf.variable_scope('conv1'):
                d_filter_conv1 = tf.get_variable('weights', [1, 15, self.depths[0], self.depths[1]], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
                outputs1 = tf.nn.conv2d(inputs, d_filter_conv1, strides=[1, 1, 1, 1], padding='SAME')
                #outputs1 = tf.layers.batch_normalization(outputs1, training=training)
                outputs1 = leaky_relu(outputs1, name='outputs')

            with tf.variable_scope('conv2'):
                d_filter_conv2 = tf.get_variable('weights', [1, 5, self.depths[1], self.depths[2]], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
                outputs2 = tf.nn.conv2d(outputs1, d_filter_conv2, strides=[1, 1, 1, 1], padding='SAME')
                #outputs2 = tf.layers.batch_normalization(outputs2, training=training)
                outputs2 = leaky_relu(outputs2, name='outputs')

            with tf.variable_scope('conv3'):
                d_filter_conv3 = tf.get_variable('weights', [1, 5, self.depths[2], self.depths[3]], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
                outputs3 = tf.nn.conv2d(outputs2, d_filter_conv3, strides=[1, 1, 5, 1], padding='SAME')
                outputs3 = tf.layers.batch_normalization(outputs3, training=training)
                outputs3 = leaky_relu(outputs3, name='outputs')

            with tf.variable_scope('conv4'):
                d_filter_conv4 = tf.get_variable('weights', [1, 5, self.depths[3], self.depths[4]], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
                outputs4 = tf.nn.conv2d(outputs3, d_filter_conv4, strides=[1, 1, 5, 1], padding='SAME')
                outputs4 = tf.layers.batch_normalization(outputs4, training=training)
                outputs4 = leaky_relu(outputs2, name='outputs')
                #print('D outputs4 shape : {}'.format(outputs4.shape))

            with tf.variable_scope('classify'):
                batch_size = inputs.get_shape()[0].value
                reshape = tf.reshape(outputs2, [batch_size, -1])
                outputs_return = tf.layers.dense(reshape, 2, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return outputs_return


class MY_GAN:
    def __init__(self,
                 batch_size=128, s_size=4, z_dim=100,
                 g_depths=[512, 2048, 4096, 2048, 4096, 2048, 512],
                 d_depths=[64, 128, 256, 512]):
        self.batch_size = batch_size
        self.s_size = s_size
        self.z_dim = z_dim
        self.g = Generator(s_size=self.s_size)
        self.d = Discriminator(depths=d_depths)
        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)

    def loss_G_only(self, noisy, train):
        noisydata = tf.convert_to_tensor(noisy)
        traindata = tf.convert_to_tensor(train)

        clears,noises = self.g(noisydata, batch_size = self.batch_size)
        clears = tf.reshape(clears, [-1,1,5000,1])
        g_outputs = self.d(clears, training=True, batch_size=self.batch_size,name='g')
        t_outputs = self.d(traindata, training=True,batch_size=self.batch_size,  name='t')

        g_loss_fit = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.ones([self.batch_size], dtype=tf.int64),
            logits = g_outputs)

        g_loss_fit = tf.reduce_mean(g_loss_fit)

        g_loss_rec = tf.reduce_mean((clears + noises - noisydata) ** 2)

        g_loss_noise = tf.reduce_mean(noises ** 2)

        g_loss = g_loss_fit * 4000 + g_loss_rec + g_loss_noise

        #return [g_loss, g_loss_fit, g_loss_rec, g_loss_noise]
        return {'g_l': g_loss, 'g_lf': g_loss_fit, 'g_lr': g_loss_rec, 'g_ln': g_loss_noise}

    def G_only_training(self, g_loss_only, g_learning_rate=0.0002, beta1=0.5):
        g_opt = tf.train.AdamOptimizer(learning_rate=g_learning_rate, beta1=beta1)
        g_opt_op = g_opt.minimize(g_loss_only['g_l'], var_list=self.g.variables)
        return g_opt_op

    def loss_D_only(self, noisy, train):
        noisydata = tf.convert_to_tensor(noisy)
        traindata = tf.convert_to_tensor(train)

        clears,noises = self.g(noisydata, batch_size = self.batch_size)
        clears = tf.reshape(clears, [-1,1,5000,1])
        g_outputs = self.d(clears, training=True, batch_size=self.batch_size,name='g')
        t_outputs = self.d(traindata, training=True,batch_size=self.batch_size,  name='t')

        g_predict = tf.nn.softmax(g_outputs)
        g_predict = tf.reduce_mean(g_predict, axis=0)
        t_predict = tf.nn.softmax(t_outputs)
        t_predict = tf.reduce_mean(t_predict, axis=0)

        d_loss_t = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.ones([self.batch_size], dtype=tf.int64),
            logits=t_outputs)

        d_loss_t = (tf.reduce_mean(d_loss_t))

        d_loss_g = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.zeros([self.batch_size], dtype=tf.int64),
            logits=g_outputs)

        d_loss_g = tf.reduce_mean(d_loss_g)

        d_loss = d_loss_g + d_loss_t

        #counting amount of D says Generated Samples "Training Samples"
        err = 0
        for i in range(64):
            tf.cond(g_outputs[i][1] > 0.5, lambda:err+1, lambda:0)
            

        return {'d_l': d_loss, 'g_pre': g_predict, 't_pre': t_predict, 'err': tf.constant(err)}

    
    def D_only_training(self, d_loss_only, d_learning_rate=0.0002, beta1=0.5):
        g_opt = tf.train.AdamOptimizer(learning_rate=d_learning_rate, beta1=beta1)        
        g_opt_op = g_opt.minimize(d_loss_only['d_l'], var_list=self.d.variables)
        return g_opt_op

    def discriminator_check(self, samples):
        samples = tf.convert_to_tensor(samples)
        outputs = self.d(samples, training=True, batch_size=self.batch_size,name='d_check')
        predictions = tf.nn.softmax(outputs)
        prediction = tf.reduce_mean(predictions, axis=0)

        return {'predictions': predictions, 'prediction': prediction}


    def sample_signals(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        clear,noise = self.g(inputs, training=True, batch_size=self.batch_size)
        clear = tf.reshape(clear, [-1])
        noise = tf.reshape(noise, [-1])
        return clear, noise

    def estimate(self, sin, clear):
        sin_tf = tf.convert_to_tensor(sin)
        clear_tf = tf.convert_to_tensor(clear)
        
        return tf.reduce_mean((sin_tf - clear_tf) ** 2)
