import tensorflow as tf
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


checkpoint_directory = "training_checkpoints_test_attention"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
num_code = 1276
batchsize = 40
max_num_visit = 200


def _array_float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _array_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))


def serialize_example(word, condition):
    feature = {
        'word': _array_int64_feature(word),
        'condition': _array_float_feature(condition)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


class Config(object):
    def __init__(self):
        self.embedding_dim = 112
        self.word_embedding_dim = 80
        self.ff_dim = 128
        self.max_num_visit = 200
        self.max_length_visit = 36
        self.vocab_dim = num_code

        self.lstm_dim = 512
        self.gap_dim = 48
        self.ar_tar = True
        self.dropconnect = True
        self.var_dropout = True
        self.n_layer = 4
        self.batchsize = batchsize


def locked_drop(inputs, is_training):
    if is_training:
        dropout_rate = 0.15
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
        self.layer_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(config.n_layer)]

    def call(self, x, is_training):
        if self.config.dropconnect:
            for layer in self.layer:
                layer.set_mask(is_training)

        for i in range(self.config.n_layer):
            x = locked_drop(x, is_training)
            x = self.layer[i](x)
            x = self.layer_norm[i](x)
        return x


class DropConnectLSTM(tf.compat.v1.keras.layers.CuDNNLSTM):
    def __init__(self, unit, return_sequences):
        super(DropConnectLSTM, self).__init__(units=unit, return_sequences=return_sequences)
        self.mask = None

    def set_mask(self,is_training):
        if is_training:
            self.mask = tf.nn.dropout(tf.ones([self.units,self.units*4]),0.3)
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


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(MultiHeadAttention, self).__init__()
        self.linear_1 = tf.keras.layers.Dense(embedding_dim,activation=tf.nn.relu)

    def call(self, inputs_tensor, is_training):
        k = self.linear_1(inputs_tensor)
        if is_training:
            k = tf.nn.dropout(k, 0.1)
        return k


class Block(tf.keras.Model):
    def __init__(self, config):
        super(Block, self).__init__()
        self.attention = MultiHeadAttention(config.embedding_dim)
        self.w1 = tf.keras.layers.Dense(config.ff_dim, activation=tf.nn.relu)
        self.w2 = tf.keras.layers.Dense(config.embedding_dim)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-8)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-8)

    def call(self, inputs, is_training):
        context = self.layer_norm1(self.attention(inputs, is_training) + inputs)
        outputs = self.layer_norm2(self.w2(self.w1(context)) + inputs)
        return outputs


class Embedding(tf.keras.Model):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.year_embed = tf.keras.layers.Embedding(27, 16)
        self.month_embed = tf.keras.layers.Embedding(12, 16)
        self.day_embed = tf.keras.layers.Embedding(31, 16)
        self.linear = tf.keras.layers.Dense(config.embedding_dim)

    def call(self, word, age):
        year = tf.one_hot(age, depth=91, dtype=tf.float32)
        word = tf.reduce_sum(tf.one_hot(word, depth=1276, dtype=tf.float32), axis=-2)
        output = self.linear(tf.concat((word, year), axis=-1))
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

    def call(self, inputs, is_training):
        outputs = self.block1(inputs, is_training)
        outputs = self.block2(outputs, is_training)
        return outputs


class Model(tf.keras.Model):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embeddings = Embedding(config)
        self.visit_att = SingleVisitTransformer(config)
        self.lstm = SingleLSTM(config)
        self.proj1 = tf.keras.layers.Dense(config.lstm_dim)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.proj2 = tf.keras.layers.Dense(256)
        self.proj3 = tf.keras.layers.Dense(config.vocab_dim)
        self.config = config

    def call(self, inputs_word, inputs_pos, inputs_gap, num_visit, is_training=False):
        inputs = self.embeddings(inputs_word, inputs_pos)
        # inputs = self.visit_att(inputs, is_training)

        gap_year = tf.cast(inputs_gap / 365, tf.int64)
        gap_month = tf.cast((inputs_gap % 365) / 31, tf.int64)
        gap_day = (inputs_gap % 365) % 31

        inputs = self.embeddings.add_gap(inputs, gap_year, gap_month, gap_day)
        inputs = self.proj1(inputs)

        output = self.lstm(inputs, is_training)
        output = self.layer_norm(self.proj2(output))
        return output



def train(config):
    feature_description = {
        'word': tf.io.FixedLenFeature([config.max_num_visit * config.max_length_visit], tf.int64),
        'pos': tf.io.FixedLenFeature([config.max_num_visit], tf.int64),
        'gap': tf.io.FixedLenFeature([config.max_num_visit], tf.int64),
        'num_visit': tf.io.FixedLenFeature([1], tf.int64),
    }

    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        word_parsed = tf.reshape(parsed['word'], [config.max_num_visit, config.max_length_visit])
        pos_parsed = tf.cast(parsed['pos'] / 365, tf.int64)

        return word_parsed, pos_parsed, parsed['gap'], parsed['num_visit']

    dataset_train = tf.data.TFRecordDataset('lstm_data/train.tfrecord')
    parsed_dataset_train = dataset_train.map(_parse_function, num_parallel_calls=4)
    parsed_dataset_train = parsed_dataset_train.batch(batchsize, drop_remainder=True).prefetch(5)

    model = Model(config)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    @tf.function
    def one_step(word_o, pos_o, gap_o, num_visit_o):
        return trns(word_o, pos_o, gap_o, num_visit_o, is_training=False)

    with tf.io.TFRecordWriter('condition_vector_2.tfrecord') as writer:
        for i,batch in enumerate(parsed_dataset_train):
            word, pos, gap, num_visit = batch
            condition_vector = one_step(word, pos, gap, num_visit)
            for b,n in enumerate(tf.squeeze(num_visit-1).numpy()):
                for k in range(n):
                    word_tmp = word[b, k+1, :].numpy()
                    condition_vector_tmp = condition_vector[b, k, :].numpy()
                    example = serialize_example(word_tmp, condition_vector_tmp)
                    writer.write(example)
            if i % 500 == 499:
                print(int(i/500))


if __name__ == '__main__':
    config = Config()
    train(config)
    # test()
