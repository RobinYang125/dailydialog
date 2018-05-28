# coding=utf-8
__author__ = 'yhd'

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.platform import gfile
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

BATCH_SIZE = 8
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_SIZE = 300
VOCAB_SIZE = 19495

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d{3,}")

import codecs

class DataLoader(object):

    def __init__(self, is_toy=False):
        if is_toy:
            self.source_train = 'data_root/train.txt'
            self.source_test = 'data_root/test.txt'
            self.batch_size = 3
            self.max_sequence_length = MAX_SEQUENCE_LENGTH
            self.source_validation = 'data_root/val.txt'
            self.test_batch_size = 3
            self.val_batch_size = 3
        else:
            self.source_train = 'data_root/data_train.txt'
            self.source_test = 'data_root/dialogues_test.txt'
            self.batch_size = BATCH_SIZE
            self.max_sequence_length = MAX_SEQUENCE_LENGTH
            self.source_validation = 'data_root/data_val.txt'
            self.test_batch_size = BATCH_SIZE
            self.val_batch_size = BATCH_SIZE

        # self.train_reader = textreader(self.source_train)
        # self.train_iterator = textiterator(self.train_reader, [self.batch_size, 2 * self.batch_size])
        #
        # self.test_reader = textreader(self.source_test)
        # self.test_iterator = textiterator(self.test_reader, [self.test_batch_size, 2 * self.test_batch_size])
        #
        # self.val_reader = textreader(self.source_validation)
        # self.val_iterator = textiterator(self.val_reader, [self.val_batch_size, 2 * self.val_batch_size])

        if platform.system() == 'Windows':
            with open(self.source_train, 'r', encoding='utf-8') as stf:
                self.train_raw_text = stf.readlines()

            with open(self.source_validation, 'r', encoding='utf-8') as svf:
                self.validation_raw_text = svf.readlines()

            with open(self.source_test, 'r', encoding='utf-8') as stef:
                self.test_raw_text = stef.readlines()

        else:
            with open(self.source_train, 'r') as stf:
                self.train_raw_text = stf.readlines()

            with open(self.source_validation, 'r') as svf:
                self.validation_raw_text = svf.readlines()

            with open(self.source_test, 'r') as stef:
                self.test_raw_text = stef.readlines()


        self.batch_num = len(self.train_raw_text) // self.batch_size
        self.val_batch_num = len(self.validation_raw_text) // self.val_batch_size
        self.test_batch_num = len(self.test_raw_text) // self.test_batch_size

        self.train_pointer = 0
        self.val_pointer = 0
        self.test_pointer = 0


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
            sentences = dialogue.split('__eou__')[:-1]
            if len(sentences) >= 3:
                for i in range(len(sentences) - 2):
                    # qa = [sentences[i], sentences[i + 1]]
                    qa_pairs.append([self.sentence_to_token_ids(sentences[i]),
                                          self.sentence_to_token_ids(sentences[i + 1]),
                                          self.sentence_to_token_ids(sentences[i + 2])])
        return qa_pairs

    def dialogues_into_qas_without_id(self, dialogues):
        qa_pairs = []
        for dialogue in dialogues:
            sentences = dialogue.split('__eou__')[:-1]
            if len(sentences) >= 3:
                for i in range(len(sentences) - 2):
                    qa = [sentences[i], sentences[i + 1], sentences[i + 2]]
                    qa_pairs.append(qa)
        return qa_pairs

    def get_batch_test(self):
        if self.test_pointer < self.test_batch_num:
            raw_data = self.test_raw_text[self.test_pointer * self.test_batch_size:
                                                (self.test_pointer + 1) * self.test_batch_size]
        else:
            raw_data = self.test_raw_text[self.test_pointer * self.test_batch_size: ]

        self.test_pointer += 1



        self.test_qa_pairs = np.asarray(self.dialogues_into_qas(raw_data))
        if self.test_qa_pairs.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None])

        self.test_raw_data = np.asarray(self.dialogues_into_qas_without_id(raw_data))
        self.test_kb_batch = self.test_raw_data[:, 0]
        self.test_q_batch = self.test_raw_data[:, 1]
        self.test_y_batch = self.test_raw_data[:, -1]

        kb_batch = self.test_qa_pairs[:, 0]
        q_batch = self.test_qa_pairs[:, 1]
        y_batch = self.test_qa_pairs[:, -1]

        kb_length = [len(item) for item in kb_batch]
        q_length = [len(item) for item in q_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]

        y_max_length = np.amax(y_length)

        return np.asarray(self.pad_sentence(kb_batch, np.amax(kb_length))), np.asarray(kb_length), \
                np.asarray(self.pad_sentence(q_batch, np.amax(q_length))), np.asarray(q_length), \
                np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length)

    def get_batch_data(self):
        if self.train_pointer < self.batch_num:
            raw_data = self.train_raw_text[self.train_pointer * self.batch_size:
                                                (self.train_pointer + 1) * self.batch_size]
        else:
            raw_data = self.train_raw_text[self.train_pointer * self.batch_size: ]

        self.train_pointer += 1

        self.qa_pairs = np.asarray(self.dialogues_into_qas(raw_data))
        if self.qa_pairs.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None])
        kb_batch = self.qa_pairs[:, 0]
        q_batch = self.qa_pairs[:, 1]
        y_batch = self.qa_pairs[:, -1]

        kb_length = [len(item) for item in kb_batch]
        q_length = [len(item) for item in q_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]

        y_max_length = np.amax(y_length)

        return np.asarray(self.pad_sentence(kb_batch, np.amax(kb_length))), np.asarray(kb_length), \
                np.asarray(self.pad_sentence(q_batch, np.amax(q_length))), np.asarray(q_length), \
               np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length)

    def get_validation(self):
        if self.val_pointer < self.val_batch_num:
            raw_data = self.validation_raw_text[self.val_pointer * self.val_batch_size:
                                                (self.val_pointer + 1) * self.val_batch_size]
        else:
            raw_data = self.validation_raw_text[self.val_pointer * self.val_batch_size: ]

        self.val_pointer += 1

        self.val_qa_pairs = np.asarray(self.dialogues_into_qas(raw_data))
        if self.val_qa_pairs.shape[0] == 0:
            return np.asarray([None]), np.asarray([None]), np.asarray([None]), np.asarray([None]), \
                   np.asarray([None]), np.asarray([None]), np.asarray([None])
        kb_batch = self.val_qa_pairs[:, 0]
        q_batch = self.val_qa_pairs[:, 1]
        y_batch = self.val_qa_pairs[:, -1]

        kb_length = [len(item) for item in kb_batch]
        q_length = [len(item) for item in q_batch]

        y_length = [len(item) + 1 for item in y_batch]

        y_max_length = np.amax(y_length)

        return np.asarray(self.pad_sentence(kb_batch, np.amax(kb_length))), np.asarray(kb_length), \
                np.asarray(self.pad_sentence(q_batch, np.amax(q_length))), np.asarray(q_length), \
               np.asarray(self.eos_pad(y_batch, y_max_length)), \
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

    def get_test_all_data(self):
        with codecs.open(self.source_test, 'r', encoding='utf-8') as test_f:
            test_data = test_f.readlines()
        test_data = np.asarray(self.dialogues_into_qas_without_id(test_data))[:, -1]
        all_test_data = []
        for line in test_data:
            all_test_data.append(line.split())
        self.all_test_data = all_test_data

    def reset_pointer(self):
        self.train_pointer = 0
        self.val_pointer = 0


class Seq2seq(object):

    def __init__(self, num_layers=1):
        self.embedding_size = EMBEDDING_SIZE
        self.vocab_size = VOCAB_SIZE
        self.num_layers = num_layers
        self.q_dimension = 2 * self.embedding_size

        self.create_model()

    def create_model(self):
        self.query_input = tf.placeholder(tf.int32, [None, None], name='query_input')
        self.query_input_lengths = tf.placeholder(tf.int32, [None], name='query_input_lengths')

        self.kb_input = tf.placeholder(tf.int32, [None, None], name='kb_input')
        self.kb_input_lengths = tf.placeholder(tf.int32, [None], name='kb_input_lengths')

        self.dropout_kp = tf.placeholder(tf.float32, name='dropout_kp', shape=())

        # GO
        self.decoder_input = tf.placeholder(tf.int32, [None, None], name='decoder_input')
        # EOS
        self.decoder_target = tf.placeholder(tf.int32, [None, None], name='decoder_target')
        self.decoder_input_lengths = tf.placeholder(tf.int32, [None], name='decoder_input_lengths')
        self.max_sequence_length = tf.reduce_max(self.decoder_input_lengths, name='max_sequence_length')
        # self.decoder_split_word = tf.placeholder(tf.int32, [None], name='decoder_split_word')

        self.mac_cell = self.create_mac_cell()
        self.mac_cell_number = 7

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.constant(0., shape=[self.vocab_size, self.embedding_size]), name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size],
                                                        name='embedding_placeholder')
            embeding_init = W.assign(self.embedding_placeholder)
            query_embedded_inputs = tf.nn.embedding_lookup(embeding_init, self.query_input)
            kb_embedded_inputs = tf.nn.embedding_lookup(embeding_init, self.kb_input)

            decoder_embedded_input = tf.nn.embedding_lookup(embeding_init, self.decoder_input)

        with tf.variable_scope('q_encoder'):
            q_fw_encoder_cells = []
            for _ in range(self.num_layers):
                q_cell = tf.contrib.rnn.GRUCell(self.embedding_size)
                q_fw_encoder_wraped_cell = tf.contrib.rnn.DropoutWrapper(q_cell, output_keep_prob=self.dropout_kp)
                q_fw_encoder_cells.append(q_fw_encoder_wraped_cell)

            q_fw_encoder_cell = tf.contrib.rnn.MultiRNNCell(q_fw_encoder_cells)

            q_bw_encoder_cells = []
            for _ in range(self.num_layers):
                q_cell = tf.contrib.rnn.GRUCell(self.embedding_size)
                q_bw_encoder_wraped_cell = tf.contrib.rnn.DropoutWrapper(q_cell, output_keep_prob=self.dropout_kp)
                q_bw_encoder_cells.append(q_bw_encoder_wraped_cell)

            q_bw_encoder_cell = tf.contrib.rnn.MultiRNNCell(q_bw_encoder_cells)

            ((query_output_fw, query_output_bw),
             (query_output_state_fw, query_output_state_bw)) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=q_fw_encoder_cell,
                                    cell_bw=q_bw_encoder_cell,
                                    inputs=query_embedded_inputs, dtype=tf.float32)

            cw = tf.concat([query_output_fw, query_output_bw], axis=-1)
            q = tf.concat([query_output_bw[:, 0], query_output_fw[:, -1]], axis=-1)

            query_state = tf.concat([query_output_state_fw, query_output_state_bw], axis=-1)

        with tf.variable_scope('kb_encoder'):
            kb_fw_encoder_cells = []
            for _ in range(self.num_layers):
                kb_cell = tf.contrib.rnn.GRUCell(self.embedding_size)
                kb_fw_encoder_wraped_cell = tf.contrib.rnn.DropoutWrapper(kb_cell, output_keep_prob=self.dropout_kp)
                kb_fw_encoder_cells.append(kb_fw_encoder_wraped_cell)

            kb_fw_encoder_cell = tf.contrib.rnn.MultiRNNCell(kb_fw_encoder_cells)

            kb_bw_encoder_cells = []
            for _ in range(self.num_layers):
                kb_cell = tf.contrib.rnn.GRUCell(self.embedding_size)
                kb_bw_encoder_wraped_cell = tf.contrib.rnn.DropoutWrapper(kb_cell, output_keep_prob=self.dropout_kp)
                kb_bw_encoder_cells.append(kb_bw_encoder_wraped_cell)

            kb_bw_encoder_cell = tf.contrib.rnn.MultiRNNCell(kb_bw_encoder_cells)


            ((kb_output_fw, kb_output_bw),
             (kb_output_state_fw, kb_output_state_bw)) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=kb_fw_encoder_cell,
                                    cell_bw=kb_bw_encoder_cell,
                                    inputs=kb_embedded_inputs, dtype=tf.float32)

            kb = tf.concat([kb_output_fw, kb_output_bw], axis=-1)

            kb_state = tf.concat([kb_output_state_fw, kb_output_state_bw], axis=-1)

        all_last_m_array = tensor_array_ops.TensorArray(dtype=tf.float32,
                                                          size=1,
                                 dynamic_size=True, infer_shape=True, clear_after_read=False)

        all_last_c_array = tensor_array_ops.TensorArray(dtype=tf.float32,
                                                          size=1,
                                 dynamic_size=True, infer_shape=True, clear_after_read=False)

        with tf.variable_scope('reasoning'):

            initial_m = tf.random_normal([tf.shape(self.query_input)[0], self.q_dimension])
            all_last_m_array = all_last_m_array.write(0, initial_m)
            initial_c = tf.random_normal([tf.shape(self.query_input)[0], self.q_dimension])
            all_last_c_array = all_last_c_array.write(0, initial_c)

            def reasoning_process(i, q, cw, kb, all_last_m, all_last_c):
                c_i, m_i = self.mac_cell(all_last_c.read(i - 1), all_last_m.read(i - 1), q, cw, kb,
                                         all_last_m, all_last_c)

                all_last_m = all_last_m.write(i, m_i)
                all_last_c = all_last_c.write(i, c_i)
                return i + 1, q, cw, kb, all_last_m, all_last_c

            i_, q_, cw_, kb_, all_last_m_array, all_last_c_array = control_flow_ops.while_loop(
                cond=lambda i, _1, _2, _3, _4, _5: i < self.mac_cell_number,
                body=reasoning_process,
                parallel_iterations=1,
                loop_vars=(tf.constant(1, dtype=tf.int32), q, cw, kb, all_last_m_array, all_last_c_array)
            )

            reasoning_output = tf.tanh(tf.concat([q_, all_last_m_array.read(i_ - 1)], axis=-1))

            middle_representation = tf.concat([kb_state[0], query_state[0], reasoning_output], axis=-1)

            reason_to_decoder_layer = Dense(self.embedding_size)
            decoder_state = reason_to_decoder_layer(middle_representation)

        with tf.variable_scope("decoder") as decoder:

            decoder_cells = []
            for _ in range(self.num_layers):
                cell = tf.contrib.rnn.GRUCell(self.embedding_size)
                decoder_wraped_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_kp)
                decoder_cells.append(decoder_wraped_cell)

            decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)

            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embedded_input,
                                                                sequence_length=self.decoder_input_lengths,
                                                                time_major=False)

            output_layer = Dense(self.vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            # 构造decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                               training_helper,
                                                               (decoder_state, ),
                                                               output_layer)
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                           impute_finished=True,
                                                    maximum_iterations=self.max_sequence_length)

        with tf.variable_scope(decoder, reuse=True):
            start_tokens = tf.tile(tf.constant([GO_ID], dtype=tf.int32), [tf.shape(self.query_input)[0]])

            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeding_init,
                                                                         start_tokens, EOS_ID)
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, predicting_helper,
                                                                 (decoder_state, ), output_layer)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                                impute_finished=False,
                                                            maximum_iterations=self.max_sequence_length)


        self.training_logits = tf.identity(training_decoder_output.rnn_output, name='training_logits')

        self.predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predicting_logits')

        masks = tf.sequence_mask(self.decoder_input_lengths, self.max_sequence_length, dtype=tf.float32, name='masks')

        self.cost = tf.contrib.seq2seq.sequence_loss(self.training_logits,
                                                     self.decoder_target, masks)

        with tf.device('/cpu:0'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            gradients = optimizer.compute_gradients(self.cost)
            capped_gradients = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gradients)


    def create_mac_cell(self):

        self.c_W_dd_i = tf.Variable(name='c_W_dd_i', dtype=tf.float32,
                                  initial_value=tf.random_normal([self.q_dimension, self.q_dimension], stddev=0.1))

        self.c_b_d_i = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[self.q_dimension]),
                                       name='c_b_d_i')

        self.c_W_2dd = tf.get_variable(name='c_W_2dd', dtype=tf.float32,
                                      shape=[2 * self.q_dimension, self.q_dimension],
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        self.c_b_d = tf.get_variable(name='c_b_d', dtype=tf.float32,
                                      shape=[self.q_dimension],
                                      initializer=tf.constant_initializer(0.1))

        self.c_W_d1 = tf.get_variable(name='c_W_d1', dtype=tf.float32,
                                      shape=[self.q_dimension, 1],
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        self.c_b_1 = tf.get_variable(name='c_b_1', dtype=tf.float32,
                                      shape=[1],
                                      initializer=tf.constant_initializer(0.1))

        self.r_W_d_d_1 = tf.get_variable(name='r_W_d_d_1', dtype=tf.float32,
                                      shape=[self.q_dimension, self.q_dimension],
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        self.r_b_d_1 = tf.get_variable(name='r_b_d_1', dtype=tf.float32,
                                      shape=[self.q_dimension],
                                      initializer=tf.constant_initializer(0.1))

        self.r_W_d_d_2 = tf.get_variable(name='r_W_d_d_2', dtype=tf.float32,
                                      shape=[self.q_dimension, self.q_dimension],
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        self.r_b_d_2 = tf.get_variable(name='r_b_d_2', dtype=tf.float32,
                                      shape=[self.q_dimension],
                                      initializer=tf.constant_initializer(0.1))

        self.r_W_2d_d_3 = tf.get_variable(name='r_W_2d_d_3', dtype=tf.float32,
                                      shape=[2 * self.q_dimension, self.q_dimension],
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        self.r_b_d_3 = tf.get_variable(name='r_b_d_3', dtype=tf.float32,
                                      shape=[self.q_dimension],
                                      initializer=tf.constant_initializer(0.1))

        self.r_W_d_1_4 = tf.get_variable(name='r_W_d_1_4', dtype=tf.float32,
                                      shape=[self.q_dimension, 1],
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        self.r_b_1_4 = tf.get_variable(name='r_b_1_4', dtype=tf.float32,
                                      shape=[1],
                                      initializer=tf.constant_initializer(0.1))

        self.w_W_2d_d_1 = tf.get_variable(name='w_W_2d_d_1', dtype=tf.float32,
                                      shape=[2 * self.q_dimension, self.q_dimension],
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        self.w_b_d_1 = tf.get_variable(name='w_b_d_1', dtype=tf.float32,
                                      shape=[self.q_dimension],
                                      initializer=tf.constant_initializer(0.1))

        self.w_W_d_1_2 = tf.get_variable(name='w_W_d_1_2', dtype=tf.float32,
                                      shape=[self.q_dimension, 1],
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        self.w_b_1_2 = tf.get_variable(name='w_b_1_2', dtype=tf.float32,
                                      shape=[1],
                                      initializer=tf.constant_initializer(0.1))

        self.w_W_2d_d_3 = tf.get_variable(name='w_W_2d_d_3', dtype=tf.float32,
                                      shape=[2 * self.q_dimension, self.q_dimension],
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        self.w_b_d_3 = tf.get_variable(name='w_b_d_3', dtype=tf.float32,
                                      shape=[self.q_dimension],
                                      initializer=tf.constant_initializer(0.1))

        self.w_W_d_d_4 = tf.get_variable(name='w_W_d_d_4', dtype=tf.float32,
                                      shape=[self.q_dimension, self.q_dimension],
                                      initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        self.w_b_d_4 = tf.get_variable(name='w_b_d_4', dtype=tf.float32,
                                      shape=[self.q_dimension],
                                      initializer=tf.constant_initializer(0.1))


        def mac_cell(last_c, last_m, q, cw, kb, all_last_m, all_last_c):
            c_i = control_unit(last_c, q, cw)
            m_new = read_unit(last_m, kb, c_i)
            m_i = write_unit(m_new, last_m, all_last_m, c_i, all_last_c)
            return c_i, m_i

        def control_unit(last_cell_state, query, cw):
            q_i = tf.matmul(query, self.c_W_dd_i) + self.c_b_d_i
            # [d]
            cq_i = tf.matmul(
                tf.concat([q_i, last_cell_state], axis=-1),
                self.c_W_2dd
            ) + self.c_b_d

            c_i = tf.reduce_mean(
                tf.nn.softmax(
                    #         [sl, d]
                    tf.tensordot(tf.expand_dims(cq_i, axis=1) * cw, self.c_W_d1, axes=[[-1], [0]]) + self.c_b_1
                ) * cw, axis=1
            )

            return c_i

        def read_unit(last_m, kb, c_i):
            m_ = tf.matmul(last_m, self.r_W_d_d_1) + self.r_b_d_1

            kb_ = tf.tensordot(kb, self.r_W_d_d_2, axes=[[-1], [0]]) + self.r_b_d_2

            i_m_kb = kb_ * tf.expand_dims(m_, axis=1)

            i_m_kb_ = tf.tensordot(tf.concat([i_m_kb, kb], axis=-1), self.r_W_2d_d_3, axes=[[-1], [0]]) + self.r_b_d_3

            # [sl,d]
            c_i_m_kb = i_m_kb_ * tf.expand_dims(c_i, axis=1)

            m_new = tf.reduce_mean(
                tf.nn.softmax(
                    tf.tensordot(c_i_m_kb, self.r_W_d_1_4, axes=[[-1], [0]]) + self.r_b_1_4
                ) * kb, axis=1
            )

            return m_new

        # TensorArray :  all_last_m  all_last_c
        def write_unit(m_new, last_m, all_last_m, c_i, all_last_c):

            all_last_m_ = all_last_m.identity()
            all_last_m_ = all_last_m_.stack()

            all_last_c_ = all_last_c.identity()
            all_last_c_ = all_last_c_.stack()

            m_ = tf.matmul(
                    tf.concat([m_new, last_m], axis=-1), self.w_W_2d_d_1
                ) + self.w_b_d_1

            m_sa = tf.reduce_mean(
                tf.nn.softmax(
                    tf.tensordot(
                        # [j-1, d]
                        tf.transpose(all_last_c_, perm=[1, 0, 2]) * tf.expand_dims(c_i, axis=1),
                        self.w_W_d_1_2, axes=[[-1], [0]]
                    ) + self.w_b_1_2
                ) * tf.transpose(all_last_m_, perm=[1, 0, 2]), axis=1
            )

            m__ = tf.matmul(tf.concat([m_sa, m_], axis=-1), self.w_W_2d_d_3) + self.w_b_d_3

            c_ = tf.matmul(c_i, self.w_W_d_d_4) + self.w_b_d_4

            m_i = tf.sigmoid(c_) * last_m + (1 - tf.sigmoid(c_)) * m__

            return m_i

        return mac_cell



    def train(self, sess, query_input, query_input_lengths, kb_input, kb_input_lengths,
              decoder_input, decoder_input_lengths,
              embedding_placeholder, dropout_kp, decoder_target):
        _, loss = sess.run([self.train_op, self.cost],
                                   feed_dict={self.query_input: query_input,
                                              self.query_input_lengths: query_input_lengths,
                                              self.kb_input: kb_input,
                                              self.kb_input_lengths: kb_input_lengths,
                                                                  self.decoder_input: decoder_input,
                                                                  self.decoder_input_lengths: decoder_input_lengths,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp,
                                                                  self.decoder_target: decoder_target})
        return loss

    def validation(self, sess, query_input, query_input_lengths, kb_input, kb_input_lengths,
                   decoder_input, decoder_input_lengths,
              embedding_placeholder, dropout_kp, decoder_target):
        loss = sess.run(self.cost,
                                   feed_dict={self.query_input: query_input,
                                              self.query_input_lengths: query_input_lengths,
                                              self.kb_input: kb_input,
                                              self.kb_input_lengths: kb_input_lengths,
                                                                  self.decoder_input: decoder_input,
                                                                  self.decoder_input_lengths: decoder_input_lengths,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp,
                                                                  self.decoder_target: decoder_target})
        return loss

    def visualization(self, sess, merged, query_input, query_input_lengths, kb_input, kb_input_lengths,
                   decoder_input, decoder_input_lengths,
              embedding_placeholder, dropout_kp, decoder_target):
        loss = sess.run(merged,
                                   feed_dict={self.query_input: query_input,
                                              self.query_input_lengths: query_input_lengths,
                                              self.kb_input: kb_input,
                                              self.kb_input_lengths: kb_input_lengths,
                                                                  self.decoder_input: decoder_input,
                                                                  self.decoder_input_lengths: decoder_input_lengths,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp,
                                                                  self.decoder_target: decoder_target})
        return loss

    def get_train_logit(self, sess, query_input, query_input_lengths, kb_input, kb_input_lengths,
                        decoder_input, decoder_input_lengths,
              embedding_placeholder, dropout_kp, decoder_target):
        logits = sess.run(self.training_logits,
                                   feed_dict={self.query_input: query_input,
                                              self.query_input_lengths: query_input_lengths,
                                              self.kb_input: kb_input,
                                              self.kb_input_lengths: kb_input_lengths,
                                                                  self.decoder_input: decoder_input,
                                                                  self.decoder_input_lengths: decoder_input_lengths,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp,
                                                                  self.decoder_target: decoder_target})
        return logits




import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug


MAX_TO_KEEP = 50

EPOCH_SIZE = 50

prefix = 'mac_reason'

def main_train(is_toy=False):
    data_loader = DataLoader(is_toy)

    model = Seq2seq()

    log_file = 'log/log_' + prefix + '.txt'
    log = codecs.open(log_file, 'w')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    print('train')
    if is_toy:
        data_loader.load_embedding('glove_false/glove.840B.300d.txt')
    else:
        data_loader.load_embedding()
    print('load the embedding matrix')

    checkpoint_storage = 'models_' + prefix + '/checkpoint'
    checkpoint_dir = 'models_' + prefix + '/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

    with tf.Session(config=config) as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=MAX_TO_KEEP)
        # merged = tf.summary.merge_all()
        # writer = tf.summary.FileWriter('summary_' + prefix + '/', sess.graph)
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

            for _ in range(data_loader.batch_num + 1):
                pad_kb_batch, kb_length, pad_q_batch, q_length, \
                    eos_pad_y_batch, go_pad_y_batch, y_length = data_loader.get_batch_data()
                if pad_kb_batch.all() == None:
                    continue
                step += 1
                loss_mean = model.train(sess, pad_q_batch, q_length, pad_kb_batch, kb_length, go_pad_y_batch,
                                        y_length, data_loader.embedding_matrix, 0.8, eos_pad_y_batch)
                losses += loss_mean

                # if step % 2 == 0:
                #     result = model.visualization(sess, merged, pad_q_batch, q_length, pad_kb_batch, kb_length, go_pad_y_batch,
                #                         y_length, data_loader.embedding_matrix, 0.8, eos_pad_y_batch)
                #     writer.add_summary(result, step)

            loss_list.append(losses / step)

            for _ in range(data_loader.val_batch_num + 1):
                pad_kb_batch, kb_length, pad_q_batch, q_length, \
                    eos_pad_y_batch, go_pad_y_batch, y_length = data_loader.get_validation()
                if pad_kb_batch.all() == None:
                    continue
                val_loss_mean = model.validation(sess, pad_q_batch, q_length, pad_kb_batch, kb_length,
                                                 go_pad_y_batch,
                                        y_length, data_loader.embedding_matrix, 1, eos_pad_y_batch)
                val_step += 1
                val_losses += val_loss_mean

            print('step', step)
            print('val_step', val_step)

            print("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g}".format(epoch + 1,
                                        EPOCH_SIZE, losses / step, val_losses / val_step))
            log.write("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g}\n".format(epoch + 1,
                                        EPOCH_SIZE, losses / step, val_losses / val_step))

            saver.save(sess, checkpoint_prefix, global_step=epoch + 1)
            print('Model Trained and Saved in epoch ', epoch + 1)

            data_loader.reset_pointer()

        # plt.plot(loss_list)
        # plt.show()

        log.close()


if platform.system() == 'Windows':
    from yhd.bleu import *
    from yhd.perplexity import *
else:
    from bleu import *
    from perplexity import *

def main_test(is_toy=False):
    data_loader = DataLoader(is_toy)
    data_loader.get_test_all_data()

    res_file = 'results/' + prefix + '_results.txt'
    res = codecs.open(res_file, 'w')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    test_graph = tf.Graph()

    with test_graph.as_default():
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # test
            print('test')
            if is_toy:
                data_loader.load_embedding('glove_false/glove.840B.300d.txt')
            else:
                data_loader.load_embedding()
            print('load the embedding matrix')

            checkpoint_file = 'models_' + prefix + '/model-50'

            loader = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            loader.restore(sess, checkpoint_file)
            print('Model has been restored')

            query_input = test_graph.get_tensor_by_name('query_input:0')
            query_input_lengths = test_graph.get_tensor_by_name('query_input_lengths:0')
            kb_input = test_graph.get_tensor_by_name('kb_input:0')
            kb_input_lengths = test_graph.get_tensor_by_name('kb_input_lengths:0')
            dropout_kp = test_graph.get_tensor_by_name('dropout_kp:0')
            decoder_input = test_graph.get_tensor_by_name('decoder_input:0')
            decoder_target = test_graph.get_tensor_by_name('decoder_target:0')
            decoder_input_lengths = test_graph.get_tensor_by_name('decoder_input_lengths:0')
            predicting_logits = test_graph.get_tensor_by_name('predicting_logits:0')
            embedding_placeholder = test_graph.get_tensor_by_name("embedding/embedding_placeholder:0")

            all_test_reply = []

            for _ in range(data_loader.test_batch_num + 1):
                pad_kb_batch, kb_length, pad_q_batch, q_length, \
                    eos_pad_y_batch, go_pad_y_batch, y_length = data_loader.get_batch_test()
                if pad_kb_batch.all() == None:
                    continue
                predicting_id = sess.run(predicting_logits,
                                       feed_dict={query_input: pad_q_batch,
                                                  query_input_lengths: q_length,
                                                  kb_input: pad_kb_batch,
                                                  kb_input_lengths: kb_length,
                                                  decoder_input: go_pad_y_batch,
                                                  decoder_input_lengths: y_length,
                                                  embedding_placeholder: data_loader.embedding_matrix,
                                                  dropout_kp: 1.0,
                                                  decoder_target: eos_pad_y_batch})

                all_reply = []
                for response in predicting_id:
                    all_reply.append([data_loader.id_vocab[id_word]
                                      for id_word in response if id_word != PAD_ID])

                all_test_reply.extend(all_reply)

                print('=========================================================')
                res.write('=========================================================\n')

                for i in range(len(data_loader.test_kb_batch)):
                    print('First Turn:')
                    res.write('First Turn:\n')
                    print(data_loader.test_kb_batch[i])
                    res.write(data_loader.test_kb_batch[i] + '\n')

                    print('Second Turn:')
                    res.write('Second Turn:\n')
                    print(data_loader.test_q_batch[i])
                    res.write(data_loader.test_q_batch[i] + '\n')

                    print('Answer:')
                    res.write('Answer:\n')
                    print(data_loader.test_y_batch[i])
                    res.write(data_loader.test_y_batch[i] + '\n')
                    print('Generation:')
                    res.write('Generation:\n')
                    print(all_reply[i])
                    res.write(' '.join(all_reply[i]) + '\n')

                    print('---------------------------------------------')
                    res.write('---------------------------------------------\n')

            ppl_input_1 = []
            for line in all_test_reply:
                ppl_input_1.append(' '.join(line))

            bleu, precisions, bp, ratio, translation_length, reference_length \
                    = compute_bleu([data_loader.all_test_data], all_test_reply, max_order=4)
            print('bleu : ', precisions)
            res.write('bleu : ' + str(precisions) + '\n')

            new_ff_test, ff_trans_cat_table, ff_train_dict = processor(
                ppl_input_1,
                data_loader.source_test
            )
            ppl = calculate_perplexity(new_ff_test, ff_trans_cat_table, ff_train_dict)
            print('perplexity : ', ppl)
            res.write('perplexity : ' + str(ppl) + '\n')

            res.close()


if __name__ == '__main__':
    # main_train(True)

    # main_test(True)

    # main_train()

    main_test()
