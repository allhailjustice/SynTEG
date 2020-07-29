import tensorflow as tf
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import gen_cudnn_rnn_ops
import time
import os
import numpy as np
from tensorflow.python.ops import array_ops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "4,6"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NUM_GPU = 2
np.set_printoptions(2,100000)
checkpoint_directory = "training_checkpoints_tpe"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
num_code = 1276
max_num_visit = 200
max_length_visit = 36
ff_dim = 256
embedding_dim = 128
lstm_dim = 512
n_layer = 2
batchsize = 2

sample_right = tf.range(2, 1001, 1,dtype=tf.float32)
sample_left = tf.range(1,1000,1,dtype=tf.float32)
sample_interval = (tf.math.log(sample_right) - tf.math.log(sample_left))

def locked_drop(inputs, is_training):
    if is_training:
        dropout_rate = 0.15
    else:
        dropout_rate = 0.0
    mask = tf.nn.dropout(tf.ones([inputs.shape[0], 1, inputs.shape[2]], dtype=tf.float32), dropout_rate)
    mask = tf.tile(mask, [1, inputs.shape[1], 1])
    return inputs * mask
    # b*t*u


class LSTM(tf.keras.Model):
    def __init__(self):
        super(LSTM, self).__init__()
        lstm = DropConnectLSTM
        self.layer = [lstm(lstm_dim, return_sequences=True) for _ in range(n_layer)]
        self.layer_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(n_layer)]

    def call(self, x, is_training):
        for layer in self.layer:
            layer.set_mask(is_training)

        for i in range(n_layer):
            x = locked_drop(x, is_training)
            x = self.layer[i](x)
            x = self.layer_norm[i](x)
        return x


class DropConnectLSTM(tf.compat.v1.keras.layers.CuDNNLSTM):
    def __init__(self, unit, return_sequences):
        super(DropConnectLSTM, self).__init__(units=unit, return_sequences=return_sequences)
        self.mask = None

    def set_mask(self, is_training):
        if is_training:
            self.mask = tf.nn.dropout(tf.ones([self.units, self.units * 4]), 0.3)
        else:
            self.mask = tf.ones([self.units, self.units * 4])

    def _process_batch(self, inputs, initial_state):
        if not self.time_major:
            inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
        input_h = initial_state[0]
        input_c = initial_state[1]
        input_h = array_ops.expand_dims(input_h, axis=0)
        input_c = array_ops.expand_dims(input_c, axis=0)

        params = recurrent_v2._canonical_to_params(  # pylint: disable=protected-access
            weights=[
                self.kernel[:, :self.units],
                self.kernel[:, self.units:self.units * 2],
                self.kernel[:, self.units * 2:self.units * 3],
                self.kernel[:, self.units * 3:],
                self.recurrent_kernel[:, :self.units] * self.mask[:, :self.units],
                self.recurrent_kernel[:, self.units:self.units * 2] * self.mask[:, self.units:self.units * 2],
                self.recurrent_kernel[:, self.units * 2:self.units * 3] * self.mask[:,
                                                                          self.units * 2:self.units * 3],
                self.recurrent_kernel[:, self.units * 3:] * self.mask[:, self.units * 3:],
            ],
            biases=[
                self.bias[:self.units],
                self.bias[self.units:self.units * 2],
                self.bias[self.units * 2:self.units * 3],
                self.bias[self.units * 3:self.units * 4],
                self.bias[self.units * 4:self.units * 5],
                self.bias[self.units * 5:self.units * 6],
                self.bias[self.units * 6:self.units * 7],
                self.bias[self.units * 7:],
            ],
            shape=self._vector_shape)

        outputs, h, c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
            inputs,
            input_h=input_h,
            input_c=input_c,
            params=params,
            is_training=True)

        if self.stateful or self.return_state:
            h = h[0]
            c = c[0]
        if self.return_sequences:
            if self.time_major:
                output = outputs
            else:
                output = array_ops.transpose(outputs, perm=(1, 0, 2))
        else:
            output = outputs[-1]
        return output, [h, c]


class Block(tf.keras.Model):
    def __init__(self):
        super(Block, self).__init__()
        self.w1 = tf.keras.layers.Dense(ff_dim, activation=tf.nn.relu)
        self.w2 = tf.keras.layers.Dense(embedding_dim)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-8)

    def call(self, inputs):
        outputs = self.layer_norm(self.w2(self.w1(inputs)) + inputs)
        return outputs


class Embedding(tf.keras.Model):
    def __init__(self):
        super(Embedding, self).__init__()
        self.age_embed = tf.keras.layers.Embedding(92, 32)
        self.year_embed = tf.keras.layers.Embedding(27, 16)
        self.month_embed = tf.keras.layers.Embedding(12, 16)
        self.day_embed = tf.keras.layers.Embedding(31, 16)
        self.linear = tf.keras.layers.Dense(embedding_dim)

    def call(self, word, age):
        year = self.age_embed(age)
        output = tf.concat((self.linear(word), year), axis=-1)
        return output

    def add_gap(self, year, month, day):
        year = self.year_embed(year)
        month = self.month_embed(month)
        day = self.day_embed(day)
        output = tf.concat((year, month, day), axis=-1)
        return output

    def no_time(self, word):
        return self.linear(word)


def abs_glorot_uniform(shape, dtype=None, partition_info=None):
    return tf.math.abs(tf.keras.initializers.glorot_uniform(seed=None)(shape,dtype=dtype))


class Transformer(tf.keras.Model):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embeddings = Embedding()
        self.lstm = LSTM()
        self.block = Block()
        self.block2 = Block()
        self.proj1 = tf.keras.layers.Dense(lstm_dim)
        self.layer_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(2)]
        self.projs = [tf.keras.layers.Dense(i, activation=tf.nn.relu) for i in [128,128]]
        self.gap_proj = tf.keras.layers.Dense(128, activation=tf.nn.tanh, kernel_initializer=abs_glorot_uniform,
                                              kernel_constraint=tf.keras.constraints.NonNeg())

        self.time_projs = [tf.keras.layers.Dense(i, activation=tf.nn.tanh, kernel_initializer=abs_glorot_uniform,
                                                 kernel_constraint=tf.keras.constraints.NonNeg()) for i in [128,128]]
        self.output_proj = tf.keras.layers.Dense(1, activation=tf.nn.softplus, kernel_initializer=abs_glorot_uniform,
                                                 kernel_constraint=tf.keras.constraints.NonNeg())

    def call(self, inputs_word, inputs_pos, inputs_gap, sequence_mask,is_training):
        inputs_word = tf.reduce_sum(tf.one_hot(inputs_word, depth=1276, dtype=tf.float32), axis=-2)
        inputs = self.embeddings(inputs_word, inputs_pos)
        # inputs = locked_drop(self.block2(locked_drop(self.block(inputs))))

        gap_year = tf.cast(inputs_gap / 365, tf.int64)
        gap_month = tf.cast((inputs_gap % 365) / 31, tf.int64)
        gap_day = (inputs_gap % 365) % 31

        gap_embed = self.embeddings.add_gap(gap_year, gap_month, gap_day)
        inputs = tf.concat((inputs,gap_embed),axis=-1)
        inputs = self.proj1(inputs)
        output = self.lstm(inputs, is_training)[:,:-1]
        # output = tf.boolean_mask(output, sequence_mask)

        # next_word = tf.boolean_mask(self.embeddings.no_time(inputs_word)[:, 1:], sequence_mask)
        next_word = self.embeddings.no_time(inputs_word)[:, 1:]
        next_word = self.block(next_word)
        if is_training:
            dropout_rate = 0.15
        else:
            dropout_rate = 0.0
        next_word = tf.nn.dropout(self.block2(tf.nn.dropout(next_word,dropout_rate)),dropout_rate)

        output = tf.concat((output, next_word), axis=-1)
        output = tf.boolean_mask(output, sequence_mask)
        for i in range(2):
            output = self.layer_norm[i](self.projs[i](output))

        gap = tf.boolean_mask(tf.cast(inputs_gap[:, 1:], tf.float32), sequence_mask)
        gap = tf.expand_dims(tf.math.log(gap + tf.random.uniform(tf.shape(gap),0,1)),-1)
        with tf.GradientTape() as g:
            g.watch(gap)
            latent = output + self.gap_proj(gap)
            for i in range(2):
                latent = self.time_projs[i](latent)
            output = self.output_proj(latent)
        hazard = g.gradient(output, gap)
        loss = -tf.reduce_mean(tf.math.log(hazard + 1e-6) - output)
        return loss

    def test(self, inputs_word, inputs_pos, inputs_gap, sequence_mask,is_training=False):
        inputs_word = tf.reduce_sum(tf.one_hot(inputs_word, depth=1276, dtype=tf.float32), axis=-2)
        inputs = self.embeddings(inputs_word, inputs_pos)
        # inputs = locked_drop(self.block2(locked_drop(self.block(inputs))))

        gap_year = tf.cast(inputs_gap / 365, tf.int64)
        gap_month = tf.cast((inputs_gap % 365) / 31, tf.int64)
        gap_day = (inputs_gap % 365) % 31

        gap_embed = self.embeddings.add_gap(gap_year, gap_month, gap_day)
        inputs = tf.concat((inputs, gap_embed), axis=-1)
        inputs = self.proj1(inputs)
        output = self.lstm(inputs, is_training)[:, :-1]
        # output = tf.boolean_mask(output, sequence_mask)

        # next_word = tf.boolean_mask(self.embeddings.no_time(inputs_word)[:, 1:], sequence_mask)
        next_word = self.embeddings.no_time(inputs_word)[:, 1:]
        next_word = self.block(next_word)
        if is_training:
            dropout_rate = 0.15
        else:
            dropout_rate = 0.0
        next_word = tf.nn.dropout(self.block2(tf.nn.dropout(next_word, dropout_rate)), dropout_rate)

        latent = tf.concat((output, next_word), axis=-1)
        latent = tf.boolean_mask(latent, sequence_mask)
        for i in range(2):
            latent = self.layer_norm[i](self.projs[i](latent))
        output = latent
        gap = tf.expand_dims(tf.math.log(tf.boolean_mask(tf.cast(inputs_gap[:, 1:], tf.float32), sequence_mask)), -1)
        with tf.GradientTape() as g:
            g.watch(gap)
            output = output + self.gap_proj(gap)
            for i in range(2):
                output = self.time_projs[i](output)
            output = self.output_proj(output)
        hazard = g.gradient(output, gap)
        loss = -tf.reduce_mean(tf.math.log(hazard + 1e-6) - output)

        # gap_test = tf.expand_dims(tf.math.log(tf.range(1,1000,dtype=tf.float32)),1)
        # output = tf.expand_dims(latent,1)
        # y = []
        # for x in output:
        #     with tf.GradientTape() as g:
        #         g.watch(gap_test)
        #         x = x + self.gap_proj(gap_test)
        #         for i in range(2):
        #             x = self.time_projs[i](x)
        #         x = tf.squeeze(-tf.math.exp(-self.output_proj(x)))
        #         # gap_test = tf.squeeze(gap_test)
        #     y.append(g.gradient(x, gap_test))
        # probs = tf.reduce_sum(tf.transpose(tf.concat(y,axis=-1)) * sample_interval,1)

        gap_test = tf.expand_dims(tf.math.log(tf.range(1,2001,dtype=tf.float32)),1)
        x = tf.expand_dims(latent,1)
        x = x + self.gap_proj(gap_test)
        for i in range(2):
            x = self.time_projs[i](x)
        x = tf.squeeze(-tf.math.exp(-self.output_proj(x)))
        probs = tf.reduce_sum(x[:,1:]-x[:,:-1],axis=-1)-tf.reduce_sum(x[:,1:100]-x[:,:99],axis=-1)
        # probs = (x[:, 1:] - x[:, :-1])[:, :100]
        return gap, loss, probs


def train():
    feature_description = {
        'word': tf.io.FixedLenFeature([max_num_visit * max_length_visit], tf.int64),
        'pos': tf.io.FixedLenFeature([max_num_visit], tf.int64),
        'gap': tf.io.FixedLenFeature([max_num_visit], tf.int64),
        'num_visit': tf.io.FixedLenFeature([1], tf.int64),
        'visit_length': tf.io.FixedLenFeature([max_num_visit], tf.int64),
    }

    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        word = tf.reshape(parsed['word'], [max_num_visit, max_length_visit])[:,1:]
        pos = tf.cast(parsed['pos'] / 365, tf.int64)
        return word, pos, parsed['gap'], parsed['num_visit']

    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

    with strategy.scope():
        dataset_train = tf.data.TFRecordDataset('lstm_data/train.tfrecord')
        parsed_dataset_train = dataset_train.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        parsed_dataset_train = parsed_dataset_train.batch(batchsize * NUM_GPU, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        dist_dataset_train = strategy.experimental_distribute_dataset(parsed_dataset_train)

        dataset_test = tf.data.TFRecordDataset('lstm_data/test.tfrecord')
        parsed_dataset_test = dataset_test.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        parsed_dataset_test = parsed_dataset_test.batch(batchsize * NUM_GPU, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        dist_dataset_test = strategy.experimental_distribute_dataset(parsed_dataset_test)

        optimizer = tf.keras.optimizers.Adam(1e-5)
        trns = Transformer()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=trns)

        def one_step(batch, is_training):
            word, pos, gap, num_visit = batch
            num_visit = tf.squeeze(num_visit)
            with tf.GradientTape() as tape:
                sequence_mask = tf.sequence_mask(num_visit - 1, max_num_visit - 1)
                gap_loss = tf.squeeze(trns(word, pos, gap, sequence_mask, is_training))
                year_loss = tf.reduce_mean((trns.embeddings.year_embed(tf.range(2, 27, dtype=tf.int64)) -
                                            trns.embeddings.year_embed(tf.range(1, 26, dtype=tf.int64))) ** 2)
                month_loss = tf.reduce_mean((trns.embeddings.month_embed(
                    tf.range(2, 12, dtype=tf.int64)) - trns.embeddings.month_embed(
                    tf.range(1, 11, dtype=tf.int64))) ** 2)
                day_loss = tf.reduce_mean((trns.embeddings.day_embed(
                    tf.range(2, 31, dtype=tf.int64)) - trns.embeddings.day_embed(tf.range(1, 30, dtype=tf.int64))) ** 2)
                loss = gap_loss + 100 * (year_loss+month_loss+day_loss)
            if is_training:
                grads = tape.gradient(loss, trns.trainable_variables)
                optimizer.apply_gradients(zip(grads, trns.trainable_variables))
            return gap_loss

        @tf.function
        def function_wrap(batch, is_training):
            return strategy.experimental_run_v2(one_step, (batch, is_training))

        def distributed_train_epoch(ds, is_training):
            total_time_loss = 0.0
            step = 0.0
            for batch in ds:
                per_replica_time_loss = function_wrap(batch, is_training)
                total_time_loss += strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_time_loss, axis=None)
                step += 1
            return total_time_loss/step

    init_time = time.time()
    print('training start')
    for epoch in range(200):
        start_time = time.time()
        with strategy.scope():
            t_loss_train = distributed_train_epoch(dist_dataset_train, True)
            t_loss_test = distributed_train_epoch(dist_dataset_test, False)

        duration_epoch = int((time.time() - start_time)/60)
        duration = int((time.time() - init_time)/60)
        format_str = 'epoch: %d, train_loss_t = %f, test_loss_t = %f (%d) (%d)'
        print(format_str % (epoch,
                            t_loss_train.numpy(),
                            t_loss_test.numpy(),
                            duration_epoch, duration))
        if epoch % 5 == 4:
            checkpoint.save(file_prefix=checkpoint_prefix)


def test():
    feature_description = {
        'word': tf.io.FixedLenFeature([max_num_visit * max_length_visit], tf.int64),
        'pos': tf.io.FixedLenFeature([max_num_visit], tf.int64),
        'gap': tf.io.FixedLenFeature([max_num_visit], tf.int64),
        'num_visit': tf.io.FixedLenFeature([1], tf.int64)
    }

    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        word = tf.reshape(parsed['word'], [max_num_visit, max_length_visit])[:,1:]
        pos = tf.cast(parsed['pos'] / 365, tf.int64)
        return word, pos, parsed['gap'], parsed['num_visit']

    dataset_test = tf.data.TFRecordDataset('lstm_data/train.tfrecord')
    parsed_dataset_test = dataset_test.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    parsed_dataset_test = parsed_dataset_test.batch(batchsize * NUM_GPU, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(1e-5)
    trns = Transformer()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=trns)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory)).expect_partial()

    for x,batch in enumerate(parsed_dataset_test):
        word, pos, gap, num_visit = batch
        num_visit = tf.squeeze(num_visit)
        sequence_mask = tf.sequence_mask(num_visit - 1, max_num_visit - 1)
        gap, loss, probs = trns.test(word, pos, gap, sequence_mask)

        # print(tf.squeeze(gap).numpy())
        # print(tf.squeeze(loss).numpy())
        print(probs.numpy())
        if x > 3:
            break


if __name__ == '__main__':
    test()
    # train()
    # test_embedding()
