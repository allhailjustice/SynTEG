import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
import time
import os
import gc
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NUM_GPU = 3

# tf.config.experimental.set_memory_growth = True
checkpoint_directory = "training_checkpoints_mt_diagnosis_256"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
num_code = 1276
batchsize = 40
max_num_visit = 200


class Config(object):
    def __init__(self):
        self.embedding_dim = 112
        self.word_embedding_dim = 80
        self.attention_size = 128
        self.ff_dim = 128
        self.max_num_visit = 200
        self.max_length_visit = 36
        self.num_head = 4
        self.vocab_dim = num_code
        self.head_dim = 32

        self.lstm_dim = 512
        self.gap_dim = 48
        self.alpha = 2.0
        self.beta = 1.0
        self.ar_tar = True
        self.dropconnect = True
        self.var_dropout = True
        self.n_layer = 3
        self.batchsize = batchsize
        self.args = [-1, self.max_num_visit, self.max_length_visit, self.num_head, self.head_dim]


def locked_drop(inputs, is_training):
    if is_training:
        dropout_rate = 0.1
    else:
        dropout_rate = 0.0
    mask = tf.nn.dropout(tf.ones([inputs.shape[0],1,inputs.shape[2]],dtype=tf.float32), dropout_rate)
    mask = tf.tile(mask, [1,inputs.shape[1],1])
    return inputs*mask
    # b*t*u


class SingleLSTM(tf.keras.Model):
    def __init__(self, config):
        super(SingleLSTM, self).__init__()
        self.config = config
        if config.dropconnect:
            lstm = DropConnectLSTM
        else:
            lstm = tf.compat.v1.keras.layers.CuDNNLSTM
        self.layer = [lstm(config.lstm_dim, return_sequences=True) for _ in range(config.n_layer)]
        self.linear = [tf.keras.layers.Dense(config.lstm_dim) for _ in range(config.n_layer)]

    def call(self, x, lengths, is_training):
        if self.config.dropconnect:
            for layer in self.layer:
                layer.set_mask(is_training)

        for i in range(self.config.n_layer):
            x0 = self.layer[i](x)
            if self.config.var_dropout and i != self.config.n_layer-1:
                x1 = locked_drop(x0, is_training)
            else:
                x1 = x0
            x2 = self.linear[i](x1)
            x = x2
        if self.config.ar_tar:
            tar = tf.nn.l2_loss(x0[:,1:,] - x0[:,:-1,]) * self.config.beta / tf.cast(self.config.lstm_dim,tf.float32) / tf.reduce_sum(tf.cast(lengths,tf.float32))
            ar = tf.nn.l2_loss(x1) * self.config.alpha / tf.cast(self.config.lstm_dim,tf.float32) / tf.reduce_sum(tf.cast(lengths,tf.float32))
            return x, ar+tar
        else:
            return x, 0


class DropConnectLSTM(tf.compat.v1.keras.layers.CuDNNLSTM):
    def __init__(self, unit, return_sequences):
        super(DropConnectLSTM, self).__init__(units=unit, return_sequences=return_sequences)
        self.mask = None

    def set_mask(self,is_training):
        if is_training:
            self.mask = tf.nn.dropout(tf.ones([self.units,self.units*4]),0.1)
        else:
            self.mask = tf.ones([self.units,self.units*4])

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
                self.recurrent_kernel[:, :self.units]*self.mask[:, :self.units],
                self.recurrent_kernel[:, self.units:self.units * 2]*self.mask[:, self.units:self.units * 2],
                self.recurrent_kernel[:, self.units * 2:self.units * 3]*self.mask[:, self.units * 2:self.units * 3],
                self.recurrent_kernel[:, self.units * 3:]*self.mask[:, self.units * 3:],
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


def reshape_to_matrix(input_tensor):
    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def transpose_for_scores(input_tensor, *args):
    output_tensor = tf.reshape(input_tensor, [*args])
    dims = list(range(len([*args])))
    dims[-2], dims[-3] = dims[-3], dims[-2]
    output_tensor = tf.transpose(output_tensor, dims)
    return output_tensor


def attention_mask(lengths, max_length_visit):
    mask = tf.expand_dims(tf.logical_not(tf.sequence_mask(lengths, max_length_visit)), -2)
    mask = tf.tile(mask, [1, 1, max_length_visit, 1])
    mask = tf.expand_dims(mask, 2)
    mask = tf.where(mask, tf.ones_like(mask, dtype=tf.float32) * tf.float32.max, tf.cast(mask, tf.float32))
    return mask


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, embedding_dim, *args):
        super(MultiHeadAttention, self).__init__()
        self.linear_wk = tf.keras.layers.Dense(args[-1] * args[-2])
        self.linear_wq = tf.keras.layers.Dense(args[-1] * args[-2])
        self.linear_wv = tf.keras.layers.Dense(args[-1] * args[-2])
        self.linear = tf.keras.layers.Dense(embedding_dim, use_bias=False)
        self.args = args

    def call(self, inputs_tensor, mask, is_training):
        k = transpose_for_scores(self.linear_wk(reshape_to_matrix(inputs_tensor)), *self.args)
        q = transpose_for_scores(self.linear_wq(reshape_to_matrix(inputs_tensor)), *self.args)
        v = transpose_for_scores(self.linear_wv(reshape_to_matrix(inputs_tensor)), *self.args)

        att = tf.matmul(q, k, transpose_b=True) / (self.args[-1] ** 0.5)
        att = tf.nn.softmax(att - mask)
        if is_training:
            att = tf.nn.dropout(att, 0.1)
        att = tf.matmul(att, v)

        dims = list(range(len(self.args)))
        dims[-2], dims[-3] = dims[-3], dims[-2]

        att = tf.transpose(att, dims)
        att = tf.reshape(att, self.args[:-2] + (self.args[-1] * self.args[-2],))
        outputs = self.linear(att)
        return outputs


class Block(tf.keras.Model):
    def __init__(self, config):
        super(Block, self).__init__()
        self.attention = MultiHeadAttention(config.embedding_dim, *config.args)
        self.w1 = tf.keras.layers.Dense(config.ff_dim, activation=tf.nn.relu)
        self.w2 = tf.keras.layers.Dense(config.embedding_dim)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-8)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-8)

    def call(self, inputs, mask, is_training):
        context = self.layer_norm1(self.attention(inputs, mask, is_training) + inputs)
        outputs = self.layer_norm2(self.w2(self.w1(context)) + inputs)
        return outputs


class Embedding(tf.keras.Model):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.word_embed = tf.keras.layers.Embedding(config.vocab_dim + 2, config.word_embedding_dim)
        self.age_embed = tf.keras.layers.Embedding(93, 32)
        self.year_embed = tf.keras.layers.Embedding(28, 16)
        self.month_embed = tf.keras.layers.Embedding(13, 16)
        self.day_embed = tf.keras.layers.Embedding(32, 16)

    def call(self, word, age):
        year = self.age_embed(age)
        word = self.word_embed(word)
        output = tf.concat((word, year), axis=-1)
        return output

    def add_gap(self, latent, year, month, day):
        year = self.year_embed(year)
        month = self.month_embed(month)
        day = self.day_embed(day)
        output = tf.concat((latent, year, month, day), axis=-1)
        return output


class SingleVisitTransformer(tf.keras.Model):
    def __init__(self, config):
        super(SingleVisitTransformer, self).__init__()
        self.block1 = Block(config)
        self.block2 = Block(config)
        self.config = config

    def call(self, inputs, visit_lengths, is_training):
        mask = attention_mask(visit_lengths, self.config.max_length_visit)
        outputs = self.block1(inputs, mask, is_training)
        outputs = self.block2(outputs, mask, is_training)
        visit_representation = outputs[:,:,0,:]
        return visit_representation


class Transformer(tf.keras.Model):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embeddings = Embedding(config)
        self.visit_att = SingleVisitTransformer(config)
        self.lstm = SingleLSTM(config)
        self.proj1 = tf.keras.layers.Dense(config.lstm_dim)
        self.proj2 = tf.keras.layers.Dense(256)
        self.proj3 = tf.keras.layers.Dense(config.vocab_dim)
        self.config = config

    def call(self, inputs_word, inputs_pos, inputs_gap, num_visit, visit_lengths, is_training=False):
        inputs = self.embeddings(inputs_word, inputs_pos)
        inputs = self.visit_att(inputs, visit_lengths, is_training)

        gap_year = tf.cast(inputs_gap / 365, tf.int64)
        gap_month = tf.cast((inputs_gap % 365) / 31, tf.int64)
        gap_day = (inputs_gap % 365) % 31

        inputs = self.embeddings.add_gap(inputs, gap_year, gap_month, gap_day)
        inputs = self.proj1(inputs)
        output, reg_loss = self.lstm(inputs, num_visit, is_training)
        output = self.proj3(self.proj2(output))

        diagnosis_output = tf.nn.sigmoid(output[:, :-1, :])
        return diagnosis_output


def cal_loss(word, num_visit, diagnosis_output):
    label = tf.reduce_sum(tf.one_hot(tf.cast(word, tf.int32), num_code, dtype=tf.float32), axis=-2)
    mask_diagnosis = tf.expand_dims(tf.squeeze(tf.cast(tf.sequence_mask(num_visit, max_num_visit-1),
                                                       tf.float32)), axis=-1)

    outputs = diagnosis_output * mask_diagnosis
    diagnosis_loss = tf.reduce_sum(tf.keras.backend.binary_crossentropy(
        label, outputs)) / tf.cast(tf.reduce_sum(num_visit), tf.float32)

    return diagnosis_loss


def train(config):
    feature_description = {
        'word': tf.io.FixedLenFeature([config.max_num_visit * config.max_length_visit], tf.int64),
        'pos': tf.io.FixedLenFeature([config.max_num_visit], tf.int64),
        'gap': tf.io.FixedLenFeature([config.max_num_visit], tf.int64),
        'num_visit': tf.io.FixedLenFeature([1], tf.int64),
        'visit_length': tf.io.FixedLenFeature([config.max_num_visit], tf.int64),
    }

    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        word = tf.reshape(parsed['word'], [config.max_num_visit, config.max_length_visit])
        pos = tf.tile(tf.expand_dims(tf.cast(parsed['pos'] / 365, tf.int64), -1), [1, config.max_length_visit - 1])
        pos = tf.concat((-2 * tf.ones((config.max_num_visit, 1), dtype=tf.int64), pos), axis=-1)

        return word, pos, parsed['gap'], parsed['num_visit'], parsed['visit_length']

    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2"])

    with strategy.scope():
        dataset_train = tf.data.TFRecordDataset('lstm_data/train.tfrecord')
        parsed_dataset_train = dataset_train.map(_parse_function, num_parallel_calls=4)
        parsed_dataset_train = parsed_dataset_train.batch(batchsize * NUM_GPU, drop_remainder=True).prefetch(5)
        dist_dataset_train = strategy.experimental_distribute_dataset(parsed_dataset_train)

        dataset_test = tf.data.TFRecordDataset('lstm_data/test.tfrecord')
        parsed_dataset_test = dataset_test.map(_parse_function, num_parallel_calls=4)
        parsed_dataset_test = parsed_dataset_test.batch(batchsize * NUM_GPU, drop_remainder=True).prefetch(5)
        dist_dataset_test = strategy.experimental_distribute_dataset(parsed_dataset_test)

        optimizer = tf.keras.optimizers.Adam(1e-4)
        trns = Transformer(config)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=trns)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

        def one_step(batch, is_training):
            word, pos, gap, num_visit, visit_length = batch
            with tf.GradientTape() as tape:
                diagnosis_output = trns(word + 2, pos + 2, gap + 1, num_visit,visit_length, is_training)
                diagnosis_loss = cal_loss(word[:, 1:, :], num_visit - 1, diagnosis_output)
                loss = diagnosis_loss
            if is_training:
                grads = tape.gradient(loss, trns.trainable_variables)
                optimizer.apply_gradients(zip(grads, trns.trainable_variables))
            return diagnosis_loss

        @tf.function
        def function_wrap(batch, is_training):
            return strategy.experimental_run_v2(one_step, (batch, is_training))

        def distributed_train_epoch(ds, is_training=True):
            total_diagnosis_loss = 0.0
            step = 0.0
            for batch in ds:
                per_replica_diagnosis_loss = function_wrap(batch, is_training)
                total_diagnosis_loss += strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_diagnosis_loss, axis=None)
                step += 1

            return total_diagnosis_loss/step

    init_time = time.time()
    print('training start')
    for epoch in range(200):
        start_time = time.time()
        with strategy.scope():
            d_loss_train = distributed_train_epoch(dist_dataset_train)
            d_loss_test = distributed_train_epoch(dist_dataset_test, False)

        duration_epoch = int((time.time() - start_time)/60)
        duration = int((time.time() - init_time)/60)
        format_str = 'epoch: %d, train_loss_d = %f, test_loss_d = %f (%d) (%d)'
        print(format_str % (epoch, d_loss_train.numpy(),
                            d_loss_test.numpy(),
                            duration_epoch, duration))
        checkpoint.save(file_prefix=checkpoint_prefix)



if __name__ == '__main__':
    config = Config()
    train(config)
    # generate(config)
    # test(config)