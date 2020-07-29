import tensorflow as tf
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NUM_GPU = 3

checkpoint_directory = "training_checkpoints_test_attention"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
num_code = 1276
batchsize = 40
max_num_visit = 200


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


class Transformer(tf.keras.Model):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embeddings = Embedding(config)
        self.visit_att = SingleVisitTransformer(config)
        self.lstm = SingleLSTM(config)
        self.proj1 = tf.keras.layers.Dense(config.lstm_dim)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.proj2 = tf.keras.layers.Dense(256)
        self.proj3 = tf.keras.layers.Dense(config.vocab_dim)
        self.config = config

    def call(self, inputs_word, inputs_pos, inputs_gap, is_training=False):
        inputs = self.embeddings(inputs_word, inputs_pos)
        # inputs = self.visit_att(inputs, is_training)

        gap_year = tf.cast(inputs_gap / 365, tf.int64)
        gap_month = tf.cast((inputs_gap % 365) / 31, tf.int64)
        gap_day = (inputs_gap % 365) % 31

        inputs = self.embeddings.add_gap(inputs, gap_year, gap_month, gap_day)
        inputs = self.proj1(inputs)

        output = self.lstm(inputs, is_training)
        output = self.proj3(self.layer_norm(self.proj2(output)))

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
        pos = tf.cast(parsed['pos'] / 365, tf.int64)
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

        # dataset_val = tf.data.TFRecordDataset('lstm_data/train.tfrecord')
        # parsed_dataset_val = dataset_val.map(_parse_function, num_parallel_calls=4)
        # parsed_dataset_val = parsed_dataset_val.batch(batchsize * NUM_GPU, drop_remainder=True).prefetch(5)
        # dist_dataset_val = strategy.experimental_distribute_dataset(parsed_dataset_val)

        optimizer = tf.keras.optimizers.Adam(1e-4)
        trns = Transformer(config)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=trns)
        # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory)).expect_partial()
        # checkpoint.restore(checkpoint_prefix + '-7')

        def one_step(batch, is_training):
            word, pos, gap, num_visit, visit_length = batch
            with tf.GradientTape() as tape:
                diagnosis_output = trns(word, pos, gap, is_training)
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
    for epoch in range(60):
        start_time = time.time()
        with strategy.scope():
            d_loss_train = distributed_train_epoch(dist_dataset_train)
            d_loss_test = distributed_train_epoch(dist_dataset_test, False)
            # if epoch % 6 == 5:
            #     d_loss_val = distributed_train_epoch(dist_dataset_val, False)
            # else:
            #     d_loss_val = tf.constant(0,dtype=tf.float32)

        duration_epoch = int((time.time() - start_time)/60)
        duration = int((time.time() - init_time)/60)
        format_str = 'epoch: %d, train_loss_d = %f, test_loss_d = %f (%d)'
        print(format_str % (epoch, d_loss_train.numpy(),
                            d_loss_test.numpy(),
                            # d_loss_val.numpy(),
                            duration_epoch))
        checkpoint.save(file_prefix=checkpoint_prefix)


def test_embedding(config):
    fig,axs = plt.subplots(1,3,figsize=(30,10))
    def plot(a,i):
        a = a.numpy()
        a = a / np.sqrt(np.sum(a ** 2, axis=-1, keepdims=True))
        d = np.matmul(a,a.T)
        sns.heatmap(d, linewidth=0, ax=axs[i])

    trns = Transformer(config)
    checkpoint = tf.train.Checkpoint(model=trns)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory)).expect_partial()
    outputs = []
    # outputs.append(trns.embeddings.age_embed(tf.range(92,dtype=tf.int64)))
    outputs.append(trns.embeddings.year_embed(tf.range(27, dtype=tf.int64)))
    outputs.append(trns.embeddings.month_embed(tf.range(12, dtype=tf.int64)))
    outputs.append(trns.embeddings.day_embed(tf.range(31, dtype=tf.int64)))
    for i,x in enumerate(outputs):
        plot(x,i)

    plt.savefig('embedding2',dpi=100)


def test(config):
    feature_description = {
        'word': tf.io.FixedLenFeature([config.max_num_visit * config.max_length_visit], tf.int64),
        'pos': tf.io.FixedLenFeature([config.max_num_visit], tf.int64),
        'gap': tf.io.FixedLenFeature([config.max_num_visit], tf.int64),
        'num_visit': tf.io.FixedLenFeature([1], tf.int64),
    }

    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        word = tf.reshape(parsed['word'], [config.max_num_visit, config.max_length_visit])
        pos = tf.cast(parsed['pos'] / 365, tf.int64)

        return word, pos, parsed['pos'], parsed['gap'], parsed['num_visit']

    dataset_train = tf.data.TFRecordDataset('lstm_data/test.tfrecord')
    parsed_dataset_train = dataset_train.map(_parse_function, num_parallel_calls=4)
    parsed_dataset_train = parsed_dataset_train.batch(batchsize, drop_remainder=True).prefetch(5)

    trns = Transformer(config)
    checkpoint = tf.train.Checkpoint(model=trns)
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory)).expect_partial()
    checkpoint.restore(checkpoint_prefix + '-20')

    @tf.function
    def one_step(word_o, pos_o, gap_o, num_visit_o):
        return trns(word_o, pos_o, gap_o, num_visit_o, is_training=False)

    p1 = []
    p2 = []
    p3 = []
    for i, batch in enumerate(parsed_dataset_train):
        word, pos, pos_, gap, num_visit = batch
        output = one_step(word+2, pos+2, gap+1, num_visit)[:,:,69].numpy()
        p1.extend(pos.numpy()[:,:199][output > 0.8])
        p2.extend(pos.numpy()[:,:199][output > 0.7])
        p3.extend(pos.numpy()[:,:199][output > 0.6])

    np.save('p1',p1)
    np.save('p2',p2)
    np.save('p3',p3)
    plt.figure(figsize=(10,10))
    plt.hist(p1,log=True)
    plt.savefig('p1',dpi=100)
    plt.figure(figsize=(10,10))
    plt.hist(p2,log=True)
    plt.savefig('p2',dpi=100)
    plt.figure(figsize=(10,10))
    plt.hist(p3,log=True)
    plt.savefig('p3',dpi=100)


if __name__ == '__main__':
    config = Config()
    train(config)
    # test(config)
    # generate(config)
    # test_embedding(config)