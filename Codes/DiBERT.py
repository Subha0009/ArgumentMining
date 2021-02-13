import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForMaskedLM
from utils import mask_token_in_text, make_pretrain_batch, get_custom_MLMmodel
import time
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def loss_fn(logits, labels, mask_value):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    weights = tf.cast(tf.equal(labels, mask_value), tf.float32)
    return tf.reduce_sum(xentropy * weights) / (tf.reduce_sum(weights)+tf.keras.backend.epsilon())

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(["[STARTQ]", "[ENDQ]", "[URL]"])
model = get_custom_MLMmodel(len(tokenizer))
optimizer = tf.keras.optimizers.Adam(2e-5)
with open('../ETC/markers.txt') as f:
    marker_list = f.read().split('\n')
if marker_list[-1]=='':
    del marker_list[-1]
with open('../Data/DiBERTdata.txt') as f:
    lines = f.readlines()

num_steps = len(list(make_pretrain_batch(lines, int(sys.argv[1]))))
sig_dict = {'input_ids': tf.TensorSpec([None, None], tf.int32, name="input_ids"),
            'token_type_ids': tf.TensorSpec([None, None], tf.int32, name="token_type_ids"), 
            'attention_mask': tf.TensorSpec([None, None], tf.int32, name="attention_mask"),
            'labels': tf.TensorSpec([None, None], tf.int32, name="labels")}

@tf.function(input_signature=[sig_dict])
def train_fn(inputs):
    with tf.GradientTape() as tape:
        _, logits = model(inputs)
        loss = loss_fn(logits, inputs['labels'], tokenizer.mask_token_id)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

for epoch in range(3):
    batch_generator = make_pretrain_batch(lines, int(sys.argv[1]))
    pbar = tf.keras.utils.Progbar(target=num_steps,
                                  width=15, 
                                  interval=0.005)
    for i in range(num_steps):
        lines = next(batch_generator)
        masked_text = [mask_token_in_text(line, marker_list, tokenizer) for line in lines]
        inputs = tokenizer(masked_text, return_tensors='tf',padding=True, truncation=True)
        inputs['labels'] = tokenizer(lines, return_tensors='tf',padding=True, truncation=True)['input_ids']
        train_fn(inputs)
        pbar.add(1)
model.save_pretrained('../SavedModels/DiBERT')
