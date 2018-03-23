# coding=utf-8
__author__ = 'yhd'

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.platform import gfile

import numpy as np

import platform

if platform.system() == 'Windows':
    from yhd.reader import *
    from yhd.iterator import *
else:
    from reader import *
    from iterator import *

import random
import copy
import re

BATCH_SIZE = 3
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_SIZE = 300
VOCAB_SIZE = 19495

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d{3,}")

class DataLoader(object):

    def __init__(self, is_toy=False):
        if is_toy:
            self.source_train = ['data_root/train.txt']
            self.source_test = ['data_root/test.txt']
            self.batch_size = 3
            self.max_sequence_length = MAX_SEQUENCE_LENGTH
            self.source_validation = ['data_root/val.txt']
            self.test_batch_size = 3
            self.val_batch_size = 3
        else:
            self.source_train = ['data_root/data_train.txt']
            self.source_test = ['data_root/dialogues_test.txt']
            self.batch_size = BATCH_SIZE
            self.max_sequence_length = MAX_SEQUENCE_LENGTH
            self.source_validation = ['data_root/data_val.txt']
            self.test_batch_size = BATCH_SIZE
            self.val_batch_size = BATCH_SIZE

        self.train_reader = textreader(self.source_train)
        self.train_iterator = textiterator(self.train_reader, [self.batch_size, 2 * self.batch_size])

        self.test_reader = textreader(self.source_test)
        self.test_iterator = textiterator(self.test_reader, [self.test_batch_size, 2 * self.test_batch_size])

        self.val_reader = textreader(self.source_validation)
        self.val_iterator = textiterator(self.val_reader, [self.val_batch_size, 2 * self.val_batch_size])

        self.initialize_vocabulary()

    def initialize_vocabulary(self, vocabulary_path='data_root/vocab50000.in'):
      """Initialize vocabulary from file.

      We assume the vocabulary is stored one-item-per-line, so a file:
        dog
        cat
      will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
      also return the reversed-vocabulary ["dog", "cat"].

      Args:
        vocabulary_path: path to the file containing the vocabulary.

      Returns:
        a pair: the vocabulary (a dictionary mapping string to integers), and
        the reversed vocabulary (a list, which reverses the vocabulary mapping).

      Raises:
        ValueError: if the provided vocabulary_path does not exist.
      """
      if gfile.Exists(vocabulary_path):
        rev_vocab = []

        with gfile.GFile(vocabulary_path, mode="r") as f:
          rev_vocab.extend(f.readlines())

        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])

        self.vocab_id = vocab
        self.id_vocab = {v: k for k, v in vocab.items()}
        self.rev_vocab = rev_vocab

    def basic_tokenizer(self, sentence):
      """Very basic tokenizer: split the sentence into a list of tokens."""
      words = []
      for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
      return [w.lower() for w in words if w]

    def sentence_to_token_ids(self, sentence, tokenizer=None, normalize_digits=True):
      """Convert a string to list of integers representing token-ids.

      For example, a sentence "I have a dog" may become tokenized into
      ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
      "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

      Args:
        sentence: a string, the sentence to convert to token-ids.
        vocabulary: a dictionary mapping tokens to integers.
        tokenizer: a function to use to tokenize each sentence;
          if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.

      Returns:
        a list of integers, the token-ids for the sentence.
      """
      if tokenizer:
        words = tokenizer(sentence)
      else:
        words = self.basic_tokenizer(sentence)
      if not normalize_digits:
        return [self.vocab_id.get(w, UNK_ID) for w in words]
      # Normalize digits by 0 before looking words up in the vocabulary.
      sentence_ids = [self.vocab_id.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]
      return sentence_ids

    def load_embedding(self, embedding_file='glove/glove.840B.300d.txt'):
        embedding_index = {}
        f = open(embedding_file)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
        embedding_index['_PAD'] = np.zeros(300, dtype=np.float32)
        embedding_index['_GO'] = np.zeros(300, dtype=np.float32)
        embedding_index['_EOS'] = np.zeros(300, dtype=np.float32)
        lookup_table = []
        num = 0
        sorted_keys = [k for k in sorted(self.id_vocab.keys())]
        for w_id in sorted_keys:
            if self.id_vocab[w_id] in embedding_index:
                num += 1
                lookup_table.append(embedding_index[self.id_vocab[w_id]])
            else:
                lookup_table.append(embedding_index['unk'])

        f.close()
        print("Total {}/{} words vector.".format(num, len(self.id_vocab)))
        self.embedding_matrix = lookup_table

    def dialogues_into_qas(self, dialogues):
        qa_pairs = []
        for dialogue in dialogues:
            sentences = dialogue.split('__eou__')
            for i in range(len(sentences) - 2):
                qa = [sentences[i], sentences[i + 1]]
                qa_pairs.append([self.sentence_to_token_ids(sentences[i]),
                                      self.sentence_to_token_ids(sentences[i + 1])])
        return qa_pairs

    def dialogues_into_qas_without_id(self, dialogues):
        qa_pairs = []
        for dialogue in dialogues:
            sentences = dialogue.split('__eou__')
            for i in range(len(sentences) - 2):
                qa = [sentences[i], sentences[i + 1]]
                qa_pairs.append(qa)
        return qa_pairs

    def get_batch_test(self):
        raw_data = self.test_iterator.next()[0]
        self.test_raw_data = np.asarray(self.dialogues_into_qas_without_id(raw_data))
        self.test_x_batch = self.test_raw_data[:, 0]
        self.test_y_batch = self.test_raw_data[:, -1]
        self.test_qa_pairs = np.asarray(self.dialogues_into_qas(raw_data))
        x_batch = self.test_qa_pairs[:, 0]
        y_batch = self.test_qa_pairs[:, -1]

        x_length = [len(item) for item in x_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]

        y_max_length = np.amax(y_length)

        return np.asarray(self.pad_sentence(x_batch, np.amax(x_length))), \
               np.asarray(x_length), np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length)

    def get_batch_data(self):
        raw_data = self.train_iterator.next()[0]
        self.qa_pairs = np.asarray(self.dialogues_into_qas(raw_data))
        x_batch = self.qa_pairs[:, 0]
        y_batch = self.qa_pairs[:, -1]

        x_length = [len(item) for item in x_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]

        y_max_length = np.amax(y_length)

        return np.asarray(self.pad_sentence(x_batch, np.amax(x_length))), \
               np.asarray(x_length), np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length)

    def get_validation(self):
        raw_data = self.val_iterator.next()[0]
        self.val_qa_pairs = np.array(self.dialogues_into_qas(raw_data))
        x_batch = self.val_qa_pairs[:, 0]
        y_batch = self.val_qa_pairs[:, -1]

        x_length = [len(item) for item in x_batch]
        y_length = [len(item) + 1 for item in y_batch]

        y_max_length = np.amax(y_length)

        return np.asarray(self.pad_sentence(x_batch, np.amax(x_length))), \
               np.asarray(x_length), np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length)

    def go_pad(self, sentences, max_length):
        return self.pad_sentence(self.add_go(sentences), max_length)

    def eos_pad(self, sentences, max_length):
        return self.pad_sentence(self.add_eos(sentences), max_length)

    def add_eos(self, sentences):
        eos_sentences = []
        for sentence in sentences:
            new_sentence = copy.copy(sentence)
            new_sentence.append(EOS_ID)
            eos_sentences.append(new_sentence)
        return eos_sentences

    def add_go(self, sentences):
        go_sentences = []
        for sentence in sentences:
            new_sentence = copy.copy(sentence)
            new_sentence.insert(0, GO_ID)
            go_sentences.append(new_sentence)
        return go_sentences

    def pad_sentence(self, sentences, max_length):
        pad_sentences = []
        for sentence in sentences:
            if len(sentence) > max_length:
                sentence = sentence[0: max_length]
            else:
                for _ in range(len(sentence), max_length):
                    sentence.append(PAD_ID)
            pad_sentences.append(sentence)
        return pad_sentences

from tensorflow.python.ops import tensor_array_ops, control_flow_ops

class Seq2seq(object):

    def __init__(self, num_layers=2):
        self.embedding_size = EMBEDDING_SIZE
        self.vocab_size = VOCAB_SIZE
        self.num_layers = num_layers

        self.create_model()

    def create_model(self):
        self.encoder_input = tf.placeholder(tf.int32, [None, None], name='encoder_input')
        self.encoder_input_lengths = tf.placeholder(tf.int32, [None], name='encoder_input_lengths')
        self.dropout_kp = tf.placeholder(tf.float32, name='dropout_kp')
        # GO
        self.decoder_input = tf.placeholder(tf.int32, [None, None], name='decoder_input')
        # EOS
        self.decoder_target = tf.placeholder(tf.int32, [None, None], name='decoder_target')
        self.decoder_input_lengths = tf.placeholder(tf.int32, [None], name='decoder_input_lengths')
        self.max_sequence_length = tf.reduce_max(self.decoder_input_lengths, name='max_sequence_length')
        # self.decoder_split_word = tf.placeholder(tf.int32, [None], name='decoder_split_word')

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.constant(0., shape=[self.vocab_size, self.embedding_size]), name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size],
                                                        name='embedding_placeholder')
            embeding_init = W.assign(self.embedding_placeholder)
            encoder_embedded_inputs = tf.nn.embedding_lookup(embeding_init, self.encoder_input)
            decoder_embedded_input = tf.nn.embedding_lookup(embeding_init, self.decoder_input)

        with tf.variable_scope('encoder'):
            encoder_cells = []
            for _ in range(self.num_layers):
                cell = tf.contrib.rnn.GRUCell(self.embedding_size)
                encoder_wraped_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_kp)
                encoder_cells.append(encoder_wraped_cell)

            encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells)
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                    inputs=encoder_embedded_inputs, dtype=tf.float32,
                                    sequence_length=self.encoder_input_lengths)

        with tf.variable_scope("decoder") as decoder:

            decoder_cells = []
            for _ in range(self.num_layers):
                cell = tf.contrib.rnn.GRUCell(self.embedding_size)
                decoder_wraped_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_kp)
                decoder_cells.append(decoder_wraped_cell)

            decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)

            output_layer = Dense(self.vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            self.decoder_outputs_array = tensor_array_ops.TensorArray(dtype=tf.float32,
                                                                      size=self.max_sequence_length,
                                             dynamic_size=False, infer_shape=True)

            def training_decode(i, hidden_state, decoder_outputs_array):
                cell_outputs, states = decoder_cell(decoder_embedded_input[:, i, :], hidden_state)
                outputs = output_layer(cell_outputs)
                decoder_outputs_array = decoder_outputs_array.write(i, outputs)
                return i + 1, hidden_state, decoder_outputs_array

            _, _, self.decoder_outputs_array = control_flow_ops.while_loop(
                cond=lambda i, _1, _2: i < self.max_sequence_length,
                body=training_decode,
                loop_vars=(tf.constant(0, dtype=tf.int32), self.encoder_state, self.decoder_outputs_array)
            )

            training_decoder_output = self.decoder_outputs_array.stack()

        with tf.variable_scope(decoder, reuse=True):

            start_tokens = tf.tile(tf.constant([GO_ID], dtype=tf.int32), [tf.shape(self.encoder_outputs)[0]])

            self.predicting_decoder_id_array = tensor_array_ops.TensorArray(dtype=tf.int64,
                                                                      size=self.max_sequence_length,
                                             dynamic_size=False, infer_shape=True)

            def predicting_decode(i, input_data, hidden_state, decoder_ids_array):
                cell_outputs, states = decoder_cell(input_data, hidden_state)
                outputs = output_layer(cell_outputs)
                output_id = tf.argmax(outputs, axis=-1)
                decoder_ids_array = decoder_ids_array.write(i, output_id)
                next_input_data = tf.nn.embedding_lookup(self.embedding_placeholder, output_id)
                return i + 1, next_input_data, hidden_state, decoder_ids_array

            _, _, _, self.predicting_decoder_id_array = control_flow_ops.while_loop(
                cond=lambda i, _1, _2, _3: i < self.max_sequence_length,
                body=predicting_decode,
                loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.embedding_placeholder,
                                                                                  start_tokens),
                           self.encoder_state, self.predicting_decoder_id_array)
            )

            predicting_decoder_output = self.predicting_decoder_id_array.stack()

        self.training_logits = tf.identity(training_decoder_output, name='training_logits')

        self.predicting_logits = tf.identity(predicting_decoder_output, name='predicting_logits')

        masks = tf.sequence_mask(self.decoder_input_lengths, self.max_sequence_length, dtype=tf.float32, name='masks')

        self.cost = tf.contrib.seq2seq.sequence_loss(self.training_logits,
                                                     self.decoder_target, masks)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        gradients = optimizer.compute_gradients(self.cost)
        capped_gradients = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)

    def train(self, sess, encoder_input, encoder_input_lengths, decoder_input, decoder_input_lengths,
              embedding_placeholder, dropout_kp, decoder_target):
        _, loss = sess.run([self.train_op, self.cost],
                                   feed_dict={self.encoder_input: encoder_input,
                                              self.encoder_input_lengths: encoder_input_lengths,
                                                                  self.decoder_input: decoder_input,
                                                                  self.decoder_input_lengths: decoder_input_lengths,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp,
                                                                  self.decoder_target: decoder_target})
        return loss

    def validation(self, sess, encoder_input, encoder_input_lengths, decoder_input, decoder_input_lengths,
              embedding_placeholder, dropout_kp, decoder_target):
        loss = sess.run(self.cost,
                                   feed_dict={self.encoder_input: encoder_input,
                                              self.encoder_input_lengths: encoder_input_lengths,
                                                                  self.decoder_input: decoder_input,
                                                                  self.decoder_input_lengths: decoder_input_lengths,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp,
                                                                  self.decoder_target: decoder_target})
        return loss

    def get_train_logit(self, sess, encoder_input, encoder_input_lengths, decoder_input, decoder_input_lengths,
              embedding_placeholder, dropout_kp, decoder_target):
        logits = sess.run(self.training_logits,
                                   feed_dict={self.encoder_input: encoder_input,
                                              self.encoder_input_lengths: encoder_input_lengths,
                                                                  self.decoder_input: decoder_input,
                                                                  self.decoder_input_lengths: decoder_input_lengths,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp,
                                                                  self.decoder_target: decoder_target})
        return logits



import os
import codecs

import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug

log_file = 'log/log.txt'
log = codecs.open(log_file, 'w')

MAX_TO_KEEP = 50

EPOCH_SIZE = 1

def main_train(is_toy=False):
    data_loader = DataLoader(is_toy)

    model = Seq2seq()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    print('train')
    data_loader.load_embedding()
    print('load the embedding matrix')

    checkpoint_storage = 'models/checkpoint'
    checkpoint_dir = 'models/'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

    with tf.Session(config=config) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=MAX_TO_KEEP)
        sess.run(tf.global_variables_initializer())
        # if os.path.exists(checkpoint_storage):
        #     checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        #     loader = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        #     loader.restore(sess, checkpoint_file)
        #     print('Model has been restored')

        loss_list = []
        # train
        for epoch in range(EPOCH_SIZE):
            losses = 0
            step = 0
            val_losses = 0
            val_step = 0
            while True:
                try:
                    pad_x_batch, x_length, eos_pad_y_batch, go_pad_y_batch, y_length = data_loader.get_batch_data()
                    step += 1
                    loss_mean = model.train(sess, pad_x_batch, x_length, go_pad_y_batch,
                                            y_length, data_loader.embedding_matrix, 0.8, eos_pad_y_batch)
                    losses += loss_mean
                except:
                    break

            loss_list.append(losses / step)

            while True:
                try:
                    pad_x_batch, x_length, eos_pad_y_batch, go_pad_y_batch, y_length = data_loader.get_validation()
                    val_loss_mean = model.validation(sess, pad_x_batch, x_length, go_pad_y_batch,
                                            y_length, data_loader.embedding_matrix, 1, eos_pad_y_batch)
                    val_step += 1
                    val_losses += val_loss_mean
                except:
                    break

            print("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g}".format(epoch,
                                        EPOCH_SIZE, losses / step, val_losses / val_step))
            log.write("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g}\n".format(epoch,
                                        EPOCH_SIZE, losses / step, val_losses / val_step))

            saver.save(sess, checkpoint_prefix, global_step=epoch)
            print('Model Trained and Saved in epoch ', epoch)

        # plt.plot(loss_list)
        # plt.show()

        log.close()

def main_test(is_toy=False):
    data_loader = DataLoader(is_toy)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True

    test_graph = tf.Graph()

    with test_graph.as_default():
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # test
            print('test')
            data_loader.load_embedding()
            print('load the embedding matrix')

            checkpoint_file = 'models/model-0'

            loader = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            loader.restore(sess, checkpoint_file)
            print('Model has been restored')

            encoder_input = test_graph.get_tensor_by_name('encoder_input:0')
            encoder_input_lengths = test_graph.get_tensor_by_name('encoder_input_lengths:0')
            dropout_kp = test_graph.get_tensor_by_name('dropout_kp:0')
            decoder_input = test_graph.get_tensor_by_name('decoder_input:0')
            decoder_target = test_graph.get_tensor_by_name('decoder_target:0')
            decoder_input_lengths = test_graph.get_tensor_by_name('decoder_input_lengths:0')
            predicting_logits = test_graph.get_tensor_by_name('predicting_logits:0')
            embedding_placeholder = test_graph.get_tensor_by_name("embedding/embedding_placeholder:0")

            # while True:
            #     try:
            pad_x_batch, x_length, eos_pad_y_batch, go_pad_y_batch, y_length = data_loader.get_batch_test()
            predicting_id = sess.run(predicting_logits,
                                   feed_dict={encoder_input: pad_x_batch,
                                              encoder_input_lengths: x_length,
                                              decoder_input: go_pad_y_batch,
                                              decoder_input_lengths: y_length,
                                              embedding_placeholder: data_loader.embedding_matrix,
                                              dropout_kp: 1.0,
                                              decoder_target: eos_pad_y_batch})

            all_reply = []
            for response in predicting_id:
                all_reply.append([data_loader.id_vocab[id_word]
                                  for id_word in response if id_word != PAD_ID])

            print('=========================================================')

            for i in range(len(data_loader.test_x_batch)):
                print('Question:')
                print(data_loader.test_x_batch[i])
                print('Answer:')
                print(data_loader.test_y_batch[i])
                print('Generation:')
                print(all_reply[i])

                print('---------------------------------------------')

                # except:
                #     break

            log.close()


if __name__ == '__main__':
    main_train(True)

    # main_test(True)

    # main_train()
    #
    # # main_test()
