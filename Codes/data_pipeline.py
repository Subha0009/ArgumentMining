import utils
import tensorflow as tf
import glob

def load_data_from_files(dir_name):
    def example_generator():
        for file_name in glob.glob(dir_name+'/*'):
            with open(file_name) as f:
                content = f.read()
            yield tf.constant(content)
    return tf.data.Dataset.from_generator(example_generator, tf.string, tf.TensorShape([]))

def clean(text):
    pattern0 = '(\n\&gt; \*Hello[\s\S]*)|(\_|\*|\#)+'
    pattern1 = "(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*(\.html|\.htm)*"
    pattern2 = "\&gt;(.*)\n\n"
    text = tf.strings.regex_replace(text, pattern1, '[URL]')
    text = tf.strings.regex_replace(text, pattern0, '')
    text = tf.strings.regex_replace(text, pattern2, '[STARTQ]'+r'\1'+' [ENDQ] ')
    return text

def return_tokenize_fn(tokenizer, marker_file):
    with open(marker_file) as f:
        marker_list = f.read().split('\n')
    if marker_list[-1]=='':
        marker_list = marker_list[:-1]
    def mask_and_tokenize(text):
        clean_text = text.numpy().decode("utf-8")
        masked_text = utils.mask_token_in_text(clean_text, marker_list, tokenizer)
        inputs = tokenizer(masked_text, padding=True, truncation=True, return_tensors='tf')
        inputs['labels'] = tokenizer(clean_text, padding=True, truncation=True, return_tensors='tf')['input_ids']
        return (tf.squeeze(inputs['input_ids']), 
                tf.squeeze(inputs['token_type_ids']), 
                tf.squeeze(inputs['attention_mask']),
                tf.squeeze(inputs['labels']))
    return mask_and_tokenize

def create_dict_for_BERT(input_ids, token_type, attention_mask, labels):
    return {"input_ids": tf.convert_to_tensor(input_ids),
            'token_type_ids': tf.convert_to_tensor(token_type),
            "attention_mask": tf.convert_to_tensor(attention_mask),
            "labels": tf.convert_to_tensor(labels)}

def load_dataset(dir_name, tokenizer, marker_file):
    dataset =  load_data_from_files(dir_name).map(clean).map(lambda x:tf.py_function(return_tokenize_fn(tokenizer, marker_file),
                                                                                     inp=[x],
                                                                                     Tout=(tf.int32, tf.int32, tf.int32, tf.int32)), 
                                                                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def _get_example_length(i1, i2, i3, i4):
    return tf.shape(i1)[-1]

def _create_min_max_boundaries(max_length,
                               min_boundary=16,
                               boundary_scale=1.1):
    bucket_boundaries = []
    x = min_boundary
    while x < max_length:
        bucket_boundaries.append(x)
        x = max(x + 1, int(x * boundary_scale))

    buckets_min = [0] + bucket_boundaries
    buckets_max = bucket_boundaries + [max_length + 1]
    return buckets_min, buckets_max


def _batch_examples(dataset, batch_size, max_length):
    buckets_min, buckets_max = _create_min_max_boundaries(max_length)
    bucket_batch_sizes = [int(batch_size) // x for x in buckets_max]
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

    def example_to_bucket_id(i1, i2, i3, i4):
        seq_length = _get_example_length(i1, i2, i3, i4)

        conditions_c = tf.logical_and(
            tf.less_equal(buckets_min, seq_length), tf.less(seq_length,
                                                        buckets_max))
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id

    def window_size_fn(bucket_id):
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id, grouped_dataset):
        bucket_batch_size = window_size_fn(bucket_id)

        return grouped_dataset.padded_batch(bucket_batch_size, ([None],[None],[None],[None]))

    return dataset.apply(
        tf.data.experimental.group_by_window(
          key_func=example_to_bucket_id,
          reduce_func=batching_fn,
          window_size=None,
          window_size_func=window_size_fn)).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def make_dataset_for_DiBERT(tokenizer, dir_name, marker_file, maxlen, batch_size):
    return _batch_examples(load_dataset(dir_name, tokenizer, marker_file), batch_size, maxlen)
