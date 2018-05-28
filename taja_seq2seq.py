# coding=utf-8
__author__ = 'yhd'

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.platform import gfile

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
            self.topic_train = ['data_root/topic_train_toy.txt']
            self.topic_val = ['data_root/topic_val_toy.txt']
            self.topic_test = ['data_root/topic_test_toy.txt']
        else:
            self.source_train = ['data_root/data_train.txt']
            self.source_test = ['data_root/dialogues_test.txt']
            self.batch_size = BATCH_SIZE
            self.max_sequence_length = MAX_SEQUENCE_LENGTH
            self.source_validation = ['data_root/data_val.txt']
            self.test_batch_size = BATCH_SIZE
            self.val_batch_size = BATCH_SIZE
            self.topic_train = ['data_root/topic_train.txt']
            self.topic_val = ['data_root/topic_val.txt']
            self.topic_test = ['data_root/topic_test.txt']

        self.initialize_reader()

        self.initialize_vocabulary()

    def initialize_reader(self):
        self.topic_train_reader = textreader(self.topic_train)
        self.topic_train_iterator = textiterator(self.topic_train_reader, [self.batch_size, 2 * self.batch_size])

        self.topic_val_reader = textreader(self.topic_val)
        self.topic_val_iterator = textiterator(self.topic_val_reader, [self.val_batch_size, 2 * self.val_batch_size])

        self.topic_test_reader = textreader(self.topic_test)
        self.topic_test_iterator = textiterator(self.topic_test_reader, [self.test_batch_size, 2 * self.test_batch_size])

        self.train_reader = textreader(self.source_train)
        self.train_iterator = textiterator(self.train_reader, [self.batch_size, 2 * self.batch_size])

        self.test_reader = textreader(self.source_test)
        self.test_iterator = textiterator(self.test_reader, [self.test_batch_size, 2 * self.test_batch_size])

        self.val_reader = textreader(self.source_validation)
        self.val_iterator = textiterator(self.val_reader, [self.val_batch_size, 2 * self.val_batch_size])


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
        batch_sizes = []
        for dialogue in dialogues:
            sentences = dialogue.split('__eou__')
            for i in range(len(sentences) - 2):
                qa = [sentences[i], sentences[i + 1]]
                qa_pairs.append([self.sentence_to_token_ids(sentences[i]),
                                      self.sentence_to_token_ids(sentences[i + 1])])
            batch_sizes.append(len(sentences) - 2)
        return qa_pairs, batch_sizes

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
        data, batch_sizes = self.dialogues_into_qas(raw_data)
        self.test_qa_pairs = np.asarray(data)
        x_batch = self.test_qa_pairs[:, 0]
        y_batch = self.test_qa_pairs[:, -1]

        x_length = [len(item) for item in x_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]

        y_max_length = np.amax(y_length)

        topic_words = self.topic_test_iterator.next()[0]
        topic_words_ids = np.asarray(self.process_topic_words(topic_words))

        all_topic_words_ids = []

        for idx in range(len(batch_sizes)):
            for _ in range(batch_sizes[idx]):
                all_topic_words_ids.append(topic_words_ids[idx])

        return np.asarray(self.pad_sentence(x_batch, np.amax(x_length))), \
               np.asarray(x_length), np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length), \
               np.asarray(all_topic_words_ids)


    def process_topic_words(self, topic_words):
        results = []
        for line in topic_words:
            t_w = self.sentence_to_token_ids(line)
            if len(t_w) >= 10:
                results.append(t_w[:10])
            else:
                while len(t_w) < 10:
                    t_w.extend(t_w)
                results.append(t_w[:10])
        return results


    def get_batch_data(self):
        raw_data = self.train_iterator.next()[0]
        data, batch_sizes = self.dialogues_into_qas(raw_data)
        self.qa_pairs = np.asarray(data)
        x_batch = self.qa_pairs[:, 0]
        y_batch = self.qa_pairs[:, -1]

        x_length = [len(item) for item in x_batch]

        # add eos
        y_length = [len(item) + 1 for item in y_batch]

        y_max_length = np.amax(y_length)

        topic_words = self.topic_train_iterator.next()[0]
        topic_words_ids = np.asarray(self.process_topic_words(topic_words))

        all_topic_words_ids = []

        for idx in range(len(batch_sizes)):
            for _ in range(batch_sizes[idx]):
                all_topic_words_ids.append(topic_words_ids[idx])

        return np.asarray(self.pad_sentence(x_batch, np.amax(x_length))), \
               np.asarray(x_length), np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length), \
               np.asarray(all_topic_words_ids)

    def get_validation(self):
        raw_data = self.val_iterator.next()[0]
        data, batch_sizes = self.dialogues_into_qas(raw_data)
        self.val_qa_pairs = np.asarray(data)
        x_batch = self.val_qa_pairs[:, 0]
        y_batch = self.val_qa_pairs[:, -1]

        x_length = [len(item) for item in x_batch]
        y_length = [len(item) + 1 for item in y_batch]

        y_max_length = np.amax(y_length)

        topic_words = self.topic_val_iterator.next()[0]
        topic_words_ids = np.asarray(self.process_topic_words(topic_words))

        all_topic_words_ids = []

        for idx in range(len(batch_sizes)):
            for _ in range(batch_sizes[idx]):
                all_topic_words_ids.append(topic_words_ids[idx])

        return np.asarray(self.pad_sentence(x_batch, np.amax(x_length))), \
               np.asarray(x_length), np.asarray(self.eos_pad(y_batch, y_max_length)), \
               np.asarray(self.go_pad(y_batch, y_max_length)), np.asarray(y_length), \
               np.asarray(all_topic_words_ids)

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
        with codecs.open(self.source_test[0], 'r', encoding='utf-8') as test_f:
            test_data = test_f.readlines()
        test_data = np.asarray(self.dialogues_into_qas_without_id(test_data))[:, -1]
        all_test_data = []
        for line in test_data:
            all_test_data.append(line.split())
        self.all_test_data = all_test_data

K = 10
SUMMARIZER_SIZE = 300

from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tensorflow.python.ops import array_ops, math_ops


class TAJA_Seq2seq(object):

    def __init__(self, num_layers=1):
        self.embedding_size = EMBEDDING_SIZE
        self.vocab_size = VOCAB_SIZE
        self.num_layers = num_layers
        self.K = K
        self.summarizer_size = SUMMARIZER_SIZE

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
        self.max_decoder_sequence_length = tf.reduce_max(self.decoder_input_lengths, name='max_decoder_sequence_length')
        self.max_encoder_sequence_length = tf.reduce_max(self.encoder_input_lengths, name='max_encoder_sequence_length')

        self.topic_words = tf.placeholder(tf.int32, [None, None], name='topic_words')

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.constant(0., shape=[self.vocab_size, self.embedding_size]), name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size],
                                                        name='embedding_placeholder')
            embeding_init = W.assign(self.embedding_placeholder)
            encoder_embedded_inputs = tf.nn.embedding_lookup(embeding_init, self.encoder_input)
            decoder_embedded_input = tf.nn.embedding_lookup(embeding_init, self.decoder_input)
            topic_words_embedded = tf.nn.embedding_lookup(embeding_init, self.topic_words)


        with tf.variable_scope('content_encoder'):
            fw_encoder_cells = []
            for _ in range(self.num_layers):
                cell = tf.contrib.rnn.GRUCell(self.embedding_size)
                fw_encoder_wraped_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_kp)
                fw_encoder_cells.append(fw_encoder_wraped_cell)

            fw_encoder_cell = tf.contrib.rnn.MultiRNNCell(fw_encoder_cells)

            bw_encoder_cells = []
            for _ in range(self.num_layers):
                cell = tf.contrib.rnn.GRUCell(self.embedding_size)
                bw_encoder_wraped_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_kp)
                bw_encoder_cells.append(bw_encoder_wraped_cell)

            bw_encoder_cell = tf.contrib.rnn.MultiRNNCell(bw_encoder_cells)

            ((content_output_fw, content_output_bw),
             (content_output_state_fw, content_output_state_bw)) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_encoder_cell,
                                    cell_bw=bw_encoder_cell,
                                    inputs=encoder_embedded_inputs, dtype=tf.float32)

            content_outputs = tf.concat([content_output_fw, content_output_bw], axis=-1)
            content_state = tf.squeeze(
                tf.concat([content_output_state_fw, content_output_state_bw], axis=-1), axis=0
            )

        with tf.variable_scope('topic_encoder'):
            topic_fw_encoder_cells = []
            for _ in range(self.num_layers):
                cell = tf.contrib.rnn.GRUCell(self.embedding_size)
                topic_fw_encoder_wraped_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_kp)
                topic_fw_encoder_cells.append(topic_fw_encoder_wraped_cell)

            topic_fw_encoder_cell = tf.contrib.rnn.MultiRNNCell(topic_fw_encoder_cells)

            topic_bw_encoder_cells = []
            for _ in range(self.num_layers):
                cell = tf.contrib.rnn.GRUCell(self.embedding_size)
                topic_bw_encoder_wraped_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_kp)
                topic_bw_encoder_cells.append(topic_bw_encoder_wraped_cell)

            topic_bw_encoder_cell = tf.contrib.rnn.MultiRNNCell(topic_bw_encoder_cells)

            # num_topic_words = tf.tile(tf.constant([self.K], dtype=tf.int32), [tf.shape(self.topic_words)[0]])

            ((topic_output_fw, topic_output_bw),
             (topic_output_state_fw, topic_output_state_bw)) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=topic_fw_encoder_cell,
                                    cell_bw=topic_bw_encoder_cell,
                                    inputs=topic_words_embedded, dtype=tf.float32)

            topic_outputs = tf.concat([topic_output_fw, topic_output_bw], axis=-1)

        with tf.variable_scope("topic_summarizer"):
            topic_words_embedded_flatten = tf.reshape(topic_words_embedded, [-1, self.K * self.embedding_size])
            summarizer_W = tf.get_variable(name='summarizer_W',
                                           shape=[self.K * self.embedding_size, self.summarizer_size],
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            summarizer_b = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[self.summarizer_size]),
                                       name='summarizer_b')
            summarizer_vector = tf.tanh(tf.nn.xw_plus_b(topic_words_embedded_flatten, summarizer_W, summarizer_b))

        with tf.variable_scope('decoder') as decoder:
            decoder_cells = []
            for _ in range(self.num_layers):
                cell = tf.contrib.rnn.GRUCell(self.embedding_size)
                decoder_wraped_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_kp)
                decoder_cells.append(decoder_wraped_cell)

            decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)

            output_layer = Dense(self.vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                         activation=tf.nn.sigmoid)

            state_layer = Dense(self.embedding_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            self.decoder_outputs_array = tensor_array_ops.TensorArray(dtype=tf.float32,
                                                                      size=self.max_decoder_sequence_length,
                                             dynamic_size=False, infer_shape=True)

            attention_size = 10

            def content_score_mlp(hidden_state):

                content_score_W_1 = tf.get_variable(name='content_score_W_1',
                                           shape=[self.embedding_size, attention_size],
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

                content_score_W_2 = tf.get_variable(name='content_score_W_2',
                                           shape=[2 * self.embedding_size, attention_size],
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

                content_score_W_3 = tf.get_variable(name='content_score_W_3',
                                           shape=[self.summarizer_size, attention_size],
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

                content_score_v = tf.get_variable(name='content_score_v',
                                           shape=[attention_size, 1],
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

                addition = tf.tanh(
                                tf.matmul(hidden_state, content_score_W_1) +
                                tf.transpose(
                                    tf.tensordot(content_outputs, content_score_W_2, axes=[[2], [0]]),
                                    perm=[1, 0, 2]) +
                                tf.matmul(summarizer_vector, content_score_W_3)
                            )

                addition = tf.transpose(addition, perm=[1, 0, 2])

                weight = tf.tensordot(addition, content_score_v, axes=[[2], [0]])

                return weight

            def topic_score_mlp(hidden_state):

                topic_score_W_1 = tf.get_variable(name='topic_score_W_1',
                                           shape=[self.embedding_size, attention_size],
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

                topic_score_W_2 = tf.get_variable(name='topic_score_W_2',
                                           shape=[2 * self.embedding_size, attention_size],
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

                topic_score_W_3 = tf.get_variable(name='topic_score_W_3',
                                           shape=[2 * self.embedding_size, attention_size],
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

                topic_score_v = tf.get_variable(name='topic_score_v',
                                           shape=[attention_size, 1],
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

                addition = tf.tanh(
                                tf.matmul(hidden_state, topic_score_W_1) +
                                tf.transpose(
                                    tf.tensordot(topic_outputs, topic_score_W_2, axes=[[2], [0]]),
                                    perm=[1, 0, 2]) +
                                tf.matmul(content_outputs[:, -1, :], topic_score_W_3)
                            )

                addition = tf.transpose(addition, perm=[1, 0, 2])

                weight = tf.tensordot(addition, topic_score_v, axes=[[2], [0]])

                return weight

            decoder_state_size = 300

            def get_overall_state(hidden_state):
                content_weights = content_score_mlp(hidden_state)
                topic_weights = topic_score_mlp(hidden_state)

                content_attention_output = tf.reduce_sum(content_outputs * content_weights, axis=1)
                topic_attention_output = tf.reduce_sum(topic_outputs * topic_weights, axis=1)

                state_W = tf.get_variable(name='state_W',
                                           shape=[self.embedding_size, decoder_state_size],
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

                content_attention_W = tf.get_variable(name='content_attention_W',
                                           shape=[2 * self.embedding_size, decoder_state_size],
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

                topic_attention_W = tf.get_variable(name='topic_attention_W',
                                           shape=[2 * self.embedding_size, decoder_state_size],
                                           dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

                decoder_b = tf.get_variable(name='decoder_b',
                                            shape=[decoder_state_size],
                                            dtype=tf.float32,
                                            initializer=tf.constant_initializer(0.1)
                                        )

                overall_state = tf.matmul(hidden_state, state_W) + \
                                tf.matmul(content_attention_output, content_attention_W) + \
                                tf.matmul(topic_attention_output, topic_attention_W) + \
                                decoder_b

                return overall_state

            training_initial_state = state_layer(content_state)

            def training_decode(i, hidden_state, decoder_outputs_array):
                overall_state = get_overall_state(hidden_state)
                cell_outputs, states = decoder_cell(decoder_embedded_input[:, i, :], (overall_state, ))
                outputs = output_layer(cell_outputs)
                decoder_outputs_array = decoder_outputs_array.write(i, outputs)
                return i + 1, states[0], decoder_outputs_array


            _, _, self.decoder_outputs_array = control_flow_ops.while_loop(
                cond=lambda i, _1, _2: i < self.max_decoder_sequence_length,
                body=training_decode,
                loop_vars=(tf.constant(0, dtype=tf.int32), training_initial_state, self.decoder_outputs_array)
            )

            training_decoder_output = tf.transpose(self.decoder_outputs_array.stack(), perm=[1, 0, 2])


        beam_width = 5

        with tf.variable_scope(decoder, reuse=True):

            def get_final_state(state):
                final_state = tensor_array_ops.TensorArray(dtype=tf.float32, size=beam_width,
                                             dynamic_size=False, infer_shape=True)
                state_array = tf.unstack(state.cell_state[0], num=beam_width, axis=1)

                for i in range(beam_width):
                    final_state = final_state.write(i, get_overall_state(state_array[i]))
                final_state = tf.transpose(final_state.stack(), perm=[1, 0, 2])
                new_state = tf.contrib.seq2seq.BeamSearchDecoderState((final_state, ), state.log_probs,
                                                                      state.finished, state.lengths)
                return new_state

            start_tokens = tf.tile(tf.constant([GO_ID], dtype=tf.int32), [tf.shape(content_state)[0]])

            overall_state = get_overall_state(state_layer(content_state))

            decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                (overall_state, ), multiplier=beam_width)

            beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell, embedding=embeding_init, start_tokens=start_tokens,
                end_token=EOS_ID, initial_state=decoder_initial_state,
                beam_width=beam_width, output_layer=output_layer
            )

            predicted_ids = tensor_array_ops.TensorArray(dtype=tf.int32,
                                                                      size=self.max_decoder_sequence_length,
                                             dynamic_size=False, infer_shape=True)

            parent_ids = tensor_array_ops.TensorArray(dtype=tf.int32,
                                                                      size=self.max_decoder_sequence_length,
                                             dynamic_size=False, infer_shape=True)

            scores = tensor_array_ops.TensorArray(dtype=tf.float32,
                                                                      size=self.max_decoder_sequence_length,
                                             dynamic_size=False, infer_shape=True)

            initial_finished, initial_inputs, initial_state = beam_search_decoder.initialize()

            initial_final_state = get_final_state(initial_state)
            initial_sequence_lengths = array_ops.zeros_like(
                    initial_finished, dtype=tf.int32)

            num_decoder_output = tf.identity(self.max_decoder_sequence_length)

            def predicting_decode(i, input_data, hidden_state, predicted_ids, parent_ids,
                                  sequence_lengths, finished, scores):
                outputs, next_state, next_inputs, decoder_finished = beam_search_decoder.step(
                    i, input_data, hidden_state
                )

                next_finished = math_ops.logical_or(decoder_finished, finished)
                next_finished = math_ops.logical_or(
                    next_finished, i + 1 >= num_decoder_output)
                next_sequence_lengths = array_ops.where(
                    math_ops.logical_and(math_ops.logical_not(finished), next_finished),
                    array_ops.fill(array_ops.shape(sequence_lengths), i + 1), sequence_lengths)

                states = get_final_state(next_state)
                predicted_ids = predicted_ids.write(i, outputs.predicted_ids)
                parent_ids = parent_ids.write(i, outputs.parent_ids)
                scores = scores.write(i, outputs.scores)
                return i + 1, next_inputs, states, predicted_ids, parent_ids, \
                       next_sequence_lengths, next_finished, scores

            _, _next_inputs, _states, predicted_ids, parent_ids, \
                      sequence_lengths, finished, scores = control_flow_ops.while_loop(
                cond=lambda i, _1, _2, _3, _4, _5, _6, _7: i < self.max_decoder_sequence_length,
                body=predicting_decode,
                loop_vars=(tf.constant(0, dtype=tf.int32), initial_inputs,
                           initial_final_state, predicted_ids, parent_ids,
                                  initial_sequence_lengths, initial_finished, scores)
            )


            predicted_ids = predicted_ids.stack()
            parent_ids = parent_ids.stack()
            scores = scores.stack()

            final_outputs_instance = tf.contrib.seq2seq.BeamSearchDecoderOutput(scores, predicted_ids, parent_ids)

            final_outputs, final_state = beam_search_decoder.finalize(
                final_outputs_instance, _states, sequence_lengths
            )

        self.training_logits = tf.identity(training_decoder_output, name='training_logits')

        self.predicting_logits = tf.identity(final_outputs.predicted_ids, name='predicting_logits')

        masks = tf.sequence_mask(self.decoder_input_lengths, self.max_decoder_sequence_length,
                                 dtype=tf.float32, name='masks')

        self.cost = tf.contrib.seq2seq.sequence_loss(self.training_logits,
                                                     self.decoder_target, masks)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
        gradients = optimizer.compute_gradients(self.cost)
        capped_gradients = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)

    def train(self, sess, encoder_input, encoder_input_lengths, decoder_input, decoder_input_lengths,
              embedding_placeholder, dropout_kp, decoder_target, topic_words):
        _, loss = sess.run([self.train_op, self.cost],
                                   feed_dict={self.encoder_input: encoder_input,
                                              self.encoder_input_lengths: encoder_input_lengths,
                                                                  self.decoder_input: decoder_input,
                                                                  self.decoder_input_lengths: decoder_input_lengths,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp,
                                                                  self.decoder_target: decoder_target,
                                              self.topic_words: topic_words})
        return loss

    def validation(self, sess, encoder_input, encoder_input_lengths, decoder_input, decoder_input_lengths,
              embedding_placeholder, dropout_kp, decoder_target, topic_words):
        loss = sess.run(self.cost,
                                   feed_dict={self.encoder_input: encoder_input,
                                              self.encoder_input_lengths: encoder_input_lengths,
                                                                  self.decoder_input: decoder_input,
                                                                  self.decoder_input_lengths: decoder_input_lengths,
                                                                  self.embedding_placeholder: embedding_placeholder,
                                                                  self.dropout_kp: dropout_kp,
                                                                  self.decoder_target: decoder_target,
                                              self.topic_words: topic_words})
        return loss

    # def get_train_logit(self, sess, encoder_input, encoder_input_lengths, decoder_input, decoder_input_lengths,
    #           embedding_placeholder, dropout_kp, decoder_target, topic_words):
    #     logits = sess.run([self.display, self.display1, self.display2, self.display3, self.display4],
    #                                feed_dict={self.encoder_input: encoder_input,
    #                                           self.encoder_input_lengths: encoder_input_lengths,
    #                                                               self.decoder_input: decoder_input,
    #                                                               self.decoder_input_lengths: decoder_input_lengths,
    #                                                               self.embedding_placeholder: embedding_placeholder,
    #                                                               self.dropout_kp: dropout_kp,
    #                                                               self.decoder_target: decoder_target,
    #                                           self.topic_words: topic_words})
    #     return logits



import os
import codecs

import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug


MAX_TO_KEEP = 50

EPOCH_SIZE = 50

def main_train(is_toy=False):
    data_loader = DataLoader(is_toy)

    log_file = 'log/taja_new_log.txt'
    log = codecs.open(log_file, 'w')

    model = TAJA_Seq2seq()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(gpu_options=gpu_options)

    print('train')
    if is_toy:
        data_loader.load_embedding('glove_false/glove.840B.300d.txt')
    else:
        data_loader.load_embedding()
    print('load the embedding matrix')

    checkpoint_storage = 'taja_new_models/checkpoint'
    checkpoint_dir = 'taja_new_models/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
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
                    pad_x_batch, x_length, eos_pad_y_batch, go_pad_y_batch, \
                        y_length, topic_words = data_loader.get_batch_data()
                    step += 1
                    loss_mean = model.train(sess, pad_x_batch, x_length, go_pad_y_batch,
                                            y_length, data_loader.embedding_matrix, 0.8, eos_pad_y_batch, topic_words)
                    losses += loss_mean
                except:
                    break

            loss_list.append(losses / step)

            while True:
                try:

                    pad_x_batch, x_length, eos_pad_y_batch, go_pad_y_batch, \
                        y_length, topic_words = data_loader.get_validation()
                    val_loss_mean = model.validation(sess, pad_x_batch, x_length, go_pad_y_batch,
                                            y_length, data_loader.embedding_matrix, 1, eos_pad_y_batch, topic_words)
                    val_step += 1
                    val_losses += val_loss_mean

                except:
                    break

            print('step', step)
            print('val_step', val_step)

            epoch_diff = 1

            print("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g}".format(epoch + epoch_diff,
                                        EPOCH_SIZE, losses / step, val_losses / val_step))
            log.write("Epoch {:>3}/{} Training Loss {:g} - Valid Loss {:g}\n".format(epoch + epoch_diff,
                                        EPOCH_SIZE, losses / step, val_losses / val_step))

            saver.save(sess, checkpoint_prefix, global_step=epoch + epoch_diff)
            print('Model Trained and Saved in epoch ', epoch + epoch_diff)

            data_loader.initialize_reader()

        # plt.plot(loss_list)
        # plt.show()

        log.close()

from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

if platform.system() == 'Windows':
    from yhd.bleu import *
    from yhd.perplexity import *
else:
    from bleu import *
    from perplexity import *

def main_test(is_toy=False):
    data_loader = DataLoader(is_toy)
    data_loader.get_test_all_data()

    res_file = 'results/taja_new_results.txt'
    res = codecs.open(res_file, 'w')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(gpu_options=gpu_options)

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

            checkpoint_file = 'taja_new_models/model-38'

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
            topic_words = test_graph.get_tensor_by_name("topic_words:0")

            all_test_reply = []

            while True:
                try:
                    pad_x_batch, x_length, eos_pad_y_batch, go_pad_y_batch, \
                        y_length, topic_data = data_loader.get_batch_test()


                    predicting_id = sess.run(predicting_logits,
                                           feed_dict={encoder_input: pad_x_batch,
                                                      encoder_input_lengths: x_length,
                                                      decoder_input: go_pad_y_batch,
                                                      decoder_input_lengths: y_length,
                                                      embedding_placeholder: data_loader.embedding_matrix,
                                                      dropout_kp: 1.0,
                                                      decoder_target: eos_pad_y_batch,
                                                      topic_words: topic_data})

                    best_answer = np.transpose(predicting_id, [1, 0, 2])[:, :, 0].tolist()

                    all_reply = []
                    for response in best_answer:
                        all_reply.append([data_loader.id_vocab[id_word]
                                          for id_word in response if id_word != PAD_ID and id_word > 0])


                    all_test_reply.extend(all_reply)

                    print('=========================================================')
                    res.write('=========================================================\n')

                    for i in range(len(data_loader.test_x_batch)):
                        print('Question:')
                        res.write('Question:\n')
                        print(data_loader.test_x_batch[i])
                        res.write(data_loader.test_x_batch[i] + '\n')
                        print('Answer:')
                        res.write('Answer:\n')
                        print(data_loader.test_y_batch[i])
                        res.write(data_loader.test_y_batch[i] + '\n')
                        print('Generation:')
                        res.write('Question:\n')
                        print(' '.join(all_reply[i]))
                        res.write(' '.join(all_reply[i]) + '\n')

                        print('---------------------------------------------')
                        res.write('---------------------------------------------\n')

                except:
                    break

            # ppl_input_1 = []
            # for line in all_test_reply:
            #     ppl_input_1.append(' '.join(line))
            #
            # bleu, precisions, bp, ratio, translation_length, reference_length \
            #         = compute_bleu([data_loader.all_test_data], all_test_reply, max_order=4)
            # print('bleu : ', precisions)
            # res.write('bleu : ' + precisions + '\n')
            #
            # new_ff_test, ff_trans_cat_table, ff_train_dict = processor(
            #     ppl_input_1,
            #     data_loader.source_test[0]
            # )
            # ppl = calculate_perplexity(new_ff_test, ff_trans_cat_table, ff_train_dict)
            # print('perplexity : ', ppl)
            # res.write('perplexity : ' + ppl + '\n')

            res.close()


if __name__ == '__main__':
    # main_train(True)
    #
    # # main_test(True)

    # main_train()

    main_test()
