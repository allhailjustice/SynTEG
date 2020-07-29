import tensorflow as tf
import numpy as np
import time
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.config.experimental.set_memory_growth = True
checkpoint_directory = "training_checkpoints_gan_new"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

num_code = 1276
batchsize = 10000
max_num_visit = 200

Z_DIM = 128
G_DIMS = [256, 256, 512, 512, 512, 512, num_code]
D_DIMS = [256, 256, 256, 128, 128, 128]


class PointWiseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(PointWiseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.bias = self.add_variable("bias",
                                      shape=[self.num_outputs],regularizer=tf.keras.regularizers.l2(1e-5))

    def call(self, x, y):
        return x * y + self.bias


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(dim,
                                                   kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                                                   bias_regularizer=tf.keras.regularizers.l2(1e-5))
                             for dim in G_DIMS[:-1]]
        self.batch_norm_layers = [tf.keras.layers.BatchNormalization(epsilon=1e-5,center=False, scale=False)
                                  for _ in G_DIMS[:-1]]
        self.output_layer = tf.keras.layers.Dense(G_DIMS[-1], activation=tf.nn.sigmoid,
                                                  kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                                                  bias_regularizer=tf.keras.regularizers.l2(1e-5))
        self.condition_layer = [tf.keras.layers.Dense(dim,
                                                      kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                                                      bias_regularizer=tf.keras.regularizers.l2(1e-5))
                                for dim in G_DIMS[:-1]]
        self.pointwiselayer = [PointWiseLayer(dim) for dim in G_DIMS[:-1]]

    def call(self, x, condition, training):
        for i in range(len(G_DIMS[:-1])):
            h = self.dense_layers[i](x)
            x = tf.nn.relu(self.pointwiselayer[i](self.batch_norm_layers[i](h, training=training), self.condition_layer[i](condition)))
        x = self.output_layer(x)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense_layers = [tf.keras.layers.Dense(dim, activation=tf.nn.relu,
                                                   kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                                                   bias_regularizer=tf.keras.regularizers.l2(1e-5))
                             for dim in D_DIMS]
        self.layer_norm_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-5,center=False, scale=False)
                                  for _ in D_DIMS]
        self.output_layer = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                                                  bias_regularizer=tf.keras.regularizers.l2(1e-5))

        self.condition_layer = [tf.keras.layers.Dense(dim,
                                                      kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                                                      bias_regularizer=tf.keras.regularizers.l2(1e-5))
                                for dim in D_DIMS]

        self.pointwiselayer = [PointWiseLayer(dim) for dim in D_DIMS]

    def call(self, x, condition):
        a = (2 * x) ** 15
        sparsity = tf.reduce_sum(a / (a + 1), axis=-1, keepdims=True)
        x = tf.concat((x, sparsity), axis=-1)
        for i in range(len(D_DIMS)):
            x = self.dense_layers[i](x)
            x = self.pointwiselayer[i](self.layer_norm_layers[i](x), self.condition_layer[i](condition))
        x = self.output_layer(x)
        return x


def train():
    feature_description = {
        'word': tf.io.FixedLenFeature([36], tf.int64),
        'condition': tf.io.FixedLenFeature([256], tf.float32)
    }

    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return parsed['word'], parsed['condition']

    dataset_train = tf.data.TFRecordDataset('condition_vector_2.tfrecord')
    parsed_dataset_train = dataset_train.map(_parse_function, num_parallel_calls=4)
    parsed_dataset_train = parsed_dataset_train.batch(batchsize, drop_remainder=True).prefetch(5)

    generator_optimizer = tf.keras.optimizers.Adam(4e-6)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-5)

    generator = Generator()
    discriminator = Discriminator()

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator,
                                     discriminator_optimizer=discriminator_optimizer, discriminator=discriminator)
    # checkpoint.restore(checkpoint_prefix+'-13')

    @tf.function
    def d_step(word, condition):
        real = word

        z = tf.random.normal(shape=[batchsize, Z_DIM])

        epsilon = tf.random.uniform(
            shape=[batchsize, 1],
            minval=0.,
            maxval=1.)

        with tf.GradientTape() as disc_tape:
            synthetic = generator(z, condition, False)
            interpolate = real + epsilon * (synthetic - real)

            real_output = discriminator(real, condition)
            fake_output = discriminator(synthetic, condition)

            w_distance = (-tf.reduce_mean(real_output) + tf.reduce_mean(fake_output))
            with tf.GradientTape() as t:
                t.watch([interpolate, condition])
                interpolate_output = discriminator(interpolate, condition)
            w_grad = t.gradient(interpolate_output, [interpolate, condition])
            slopes = tf.sqrt(tf.reduce_sum(tf.square(w_grad[0]), 1)+tf.reduce_sum(tf.square(w_grad[1]), 1))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            reg_loss = tf.reduce_sum(discriminator.losses)
            disc_loss = 10 * gradient_penalty + w_distance + reg_loss

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return disc_loss, w_distance, reg_loss

    @tf.function
    def g_step(condition):
        z = tf.random.normal(shape=[batchsize, Z_DIM])
        with tf.GradientTape() as gen_tape:
            synthetic = generator(z, condition, True)

            fake_output = discriminator(synthetic, condition)

            gen_loss = -tf.reduce_mean(fake_output) + tf.reduce_sum(generator.losses)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    @tf.function
    def train_step(batch):
        word, condition = batch
        word = tf.reduce_sum(tf.one_hot(word, depth=num_code, dtype=tf.float32), axis=-2)
        disc_loss, w_distance, reg_loss = d_step(word, condition)
        g_step(condition)
        return disc_loss, w_distance, reg_loss

    print('training start')
    for epoch in range(2000):
        start_time = time.time()
        total_loss = 0.0
        total_w = 0.0
        total_reg = 0.0
        step = 0.0
        for args in parsed_dataset_train:
            loss, w, reg = train_step(args)
            total_loss += loss
            total_w += w
            total_reg += reg
            step += 1
        duration_epoch = time.time() - start_time
        format_str = 'epoch: %d, loss = %f, w = %f, reg = %f (%.2f)'
        print(format_str % (epoch, -total_loss / step, -total_w / step, total_reg / step, duration_epoch))
        if epoch % 50 == 49:
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    train()
