import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow_addons.text.crf import crf_log_likelihood
import numpy as np

def get_transition(num_classes, max_val=0.1, min_val=-0.1, is_padded=False):
    '''
    0:None, 1:B-Claim, 2:I-Claim, 3:B-Premise, 4:I-Premise, 5:PAD (optional)
    Creates transition matrix to initialize crf
    '''
    T = np.random.uniform(min_val, max_val, size=(num_classes, num_classes))
    T[0, [2, 4]] = [-100000., -100000.]
    T[1, [0, 1, 3, 4]] = [-100000., -100000., -100000., -100000.]
    T[2, [4]] = [-100000.]
    T[3, [0, 1, 2, 3]] = [-100000., -100000., -100000., -100000.]
    T[4, [2]] = [-100000.]
    if is_padded:
        T[5, :] = [-100000. for _ in range(num_classes)]
        T[[1, 3], 5] = [-100000., -100000.]
    return T.astype('float32')

def compute_dsc_loss(y_pred, y_true, alpha=0.6):
    y_pred = K.reshape(K.softmax(y_pred), (-1, y_pred.shape[2]))
    y = K.expand_dims(K.flatten(y_true), axis=1)
    probs = tf.gather_nd(y_pred, y, batch_dims=1)
    pos = K.pow(1 - probs, alpha) * probs
    dsc_loss = 1 - (2 * pos + 1) / (pos + 2)
    return dsc_loss

class CRF(tf.keras.layers.Layer):
    def __init__(self, transition_matrix, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.transitions = tf.Variable(transition_matrix)
        self.supports_masking = True
        self.mask = None
    def call(self, inputs, mask=None, training=None):
        if mask is None:
            raw_input_shape = tf.slice(tf.shape(inputs), [0], [2])
            mask = tf.ones(raw_input_shape)
        sequence_lengths = K.sum(K.cast(mask, 'int32'), axis=-1)

        viterbi_sequence, _ = tfa.text.crf_decode(
            inputs, self.transitions, sequence_lengths
        )
        if training:
            return viterbi_sequence, inputs, sequence_lengths, self.transitions
        else:
            return viterbi_sequence

class TaskModel(tf.keras.models.Model):
    def __init__(self, encoder, 
                 max_trans=0.5, 
                 min_trans=0., 
                 is_padded=False, 
                 use_dsc=True, 
                 alpha=0.6, lr=2e-5):
        super(TaskModel, self).__init__()
        self.encoder = encoder
        if is_padded:
            num_classes = 6
        else:
            num_classes = 5
        self.ff = tf.keras.layers.Dense(num_classes)
        self.use_dsc = use_dsc
        self.alpha = alpha
        self.crf_layer = CRF(get_transition(num_classes, max_val=max_trans, min_val=min_trans, is_padded=is_padded))
        self.optimizer = tf.keras.optimizers.Adam(lr)

    def call(self, inputs, training=True):
        encoded_seq = self.encoder(inputs, training=training)
        logits = self.ff(encoded_seq['last_hidden_state'])
        crf_predictions = self.crf_layer(logits, training=training)
        if training:
            viterbi_sequence, potentials, sequence_length, chain_kernel = crf_predictions
            return viterbi_sequence, potentials, sequence_length, chain_kernel, logits
        else:
            return crf_predictions

    def compute_loss(self, x, y, sample_weight=None, training=True):
        viterbi_sequence, potentials, sequence_length, chain_kernel, logits = self(x, training=training)
        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]
        #ds_loss = compute_dsc_loss(potentials, y, self.alpha)
        cc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits)
        if sample_weight is not None:
            cc_loss = tf.reduce_sum(cc_loss*sample_weight)/tf.reduce_sum(sample_weight)
        return tf.reduce_mean(crf_loss), cc_loss

    def compute_batch_sample_weight(self, labels):
        _, _, counts = tf.unique_with_counts(tf.reshape(labels, [-1]))
        counts = tf.cast(counts, dtype=tf.float32) + tf.keras.backend.epsilon()
        class_weights = tf.math.log(tf.reduce_sum(counts)/counts)
        non_pad = tf.cast(tf.math.not_equal(labels, 5), dtype=tf.float32)
        weighted_labels = tf.gather(class_weights, labels)
        return non_pad*weighted_labels

    def train_step(self, inputs, labels, sample_weight=None):
        with tf.GradientTape() as tape:
            crf_loss, ds_loss, _ = self.compute_loss(inputs, labels, sample_weight)
            if self.use_dsc:
                total_loss = crf_loss + ds_loss
            else:
                total_loss = crf_losss
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    '''
    @tf.function(input_signature=[(tf.TensorSpec([], tf.string, name="idx"),
                                   tf.TensorSpec([None, None], tf.int32, name="input_ids"), 
                                   tf.TensorSpec([None, None], tf.int32, name="attention_mask"),
                                   tf.TensorSpec([None, None], tf.int32, name="global_attention_mask"),
                                   tf.TensorSpec([None, None], tf.int32, name="labels"))])

    def batch_train_step(self, inputs):
        idx, input_ids, attention_mask, global_attention_mask, labels = inputs
        inputs['input_ids'] = input_ids
        inputs['attention_mask'] = attention_mask
        inputs['global_attention_mask'] = global_attention_mask
        sample_weight = self.compute_batch_sample_weight(labels)
        with tf.GradientTape() as tape:
            crf_loss, ds_loss = self.compute_loss(inputs, labels, sample_weight)
            if self.use_dsc:
                total_loss = crf_loss + ds_loss
            else:
                total_loss = crf_losss
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    '''
    def infer_step(self, inputs):
        viterbi_sequence = self(inputs, training=False)
        return viterbi_sequence
