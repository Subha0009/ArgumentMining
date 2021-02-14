import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForMaskedLM
from utils import get_custom_MLMmodel
from data_pipeline import make_dataset_for_DiBERT
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
signature = (tf.TensorSpec([None, None], tf.int32, name="input_ids"),
             tf.TensorSpec([None, None], tf.int32, name="token_type_ids"), 
             tf.TensorSpec([None, None], tf.int32, name="attention_mask"),
             tf.TensorSpec([None, None], tf.int32, name="labels"))

@tf.function(input_signature=[signature])
def train_fn(inp):
    inputs = {}
    inputs['input_ids'] = inp[0]
    inputs['token_type_ids'] = inp[1]
    inputs['attention_mask'] = inp[2]
    inputs['labels'] = inp[3]
    with tf.GradientTape() as tape:
        _, logits = model(inputs)
        loss = loss_fn(logits, inputs['labels'], tokenizer.mask_token_id)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

train_data = make_dataset_for_DiBERT(tokenizer, 
                                     sys.argv[1], 
                                     sys.argv[2], 
                                     int(sys.argv[3]),
                                     int(sys.argv[4]))
steps = 100000
for epoch in range(3):
    print('Epoch: {}'.format(epoch))
    pbar = tf.keras.utils.Progbar(target=steps,
                                  width=15, 
                                  interval=0.005)
    steps = 0
    for inputs in train_data:
        train_fn(inputs)
        pbar.add(1)
        steps +=1
    model.save_pretrained('../SavedModels/DiBERT')
