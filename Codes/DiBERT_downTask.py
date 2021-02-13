import json
import tensorflow as tf
from transformers import TFBertForMaskedLM, BertTokenizer
from utils import tokenize_and_label, train_test_split
from sklearn.metrics import classification_report

lm_model = TFBertForMaskedLM.from_pretrained('../SavedModels/DiBERT') # For fine tuned bert
#lm_model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')  For pretrained bert
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(["[STARTQ]", "[ENDQ]", "[URL]"])
lm_model.resize_token_embeddings(len(tokenizer))

class TaskModel(tf.keras.models.Model):
    def __init__(self, trained_lm_model, num_classes=5):
        super(TaskModel, self).__init__()
        self.encoder = trained_lm_model.layers[0]
        self.prediction = tf.keras.layers.Dense(num_classes, activation='softmax')
    def call(self, inputs):
        encoded_seq, _ = self.encoder(inputs)
        return self.prediction(encoded_seq)

task_model = TaskModel(lm_model)
optimizer = tf.keras.optimizers.Adam(lr=0.00001)

with open('../Data/annotated_threads.json', 'r') as f:
    data = json.load(f)
train_keys, test_keys = train_test_split()
result_file = open('BERT_results.txt', 'w')
for epoch in range(1, 6):
    for key in train_keys:
        for sample in data[key]:
            inputs, label = tokenize_and_label(sample, tokenizer)
            with tf.GradientTape() as tape:
                output = task_model(inputs)
                loss = tf.keras.losses.sparse_categorical_crossentropy(tf.convert_to_tensor(label), output)
            grads = tape.gradient(loss, task_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, task_model.trainable_variables))

    y_true, y_pred = [], []
    for key in test_keys:
        for sample in data[key]:
            inputs, label = tokenize_and_label(sample, tokenizer)
            y_true.extend(label[0])
            output = task_model(inputs)
            y_pred.extend(list(tf.argmax(output, axis=-1).numpy()[0]))
    result_file.write(classification_report(y_true, y_pred))
    print('Epoch {} done'.format(epoch))
result_file.close()
