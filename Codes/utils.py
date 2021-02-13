import json
import re
from transformers import BertForMaskedLM, TFBertForMaskedLM
from cleaner import clean_pipeline

def create_mask_map(markers, tokenizer, special_token):
    '''
    Given list of markers to mask, prepares a list of tuples 
    (marker, <space separated mask tokens equal to the number of  subwords from marker>)
    '''
    marker_map = []
    for marker in markers:
        masks = ' '.join([special_token for _ in range(len(tokenizer.encode(marker))-2)])
        marker_map.append((marker, masks))
    return sorted(marker_map, key=lambda x:len(x[1].split()), reverse=True)

def mask_token_in_text(text, markers, tokenizer):
    '''
    Mask tokens from given list of markers
    '''
    marker_map = create_mask_map(markers, tokenizer, tokenizer.mask_token)
    for marker, mask in marker_map:
        regex = r'(?<=\W)%s(?=(\s|\W|$))'%marker
        text = re.sub(regex, mask, text, flags=re.IGNORECASE)
    return text


def get_custom_MLMmodel(num_tokens):
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.resize_token_embeddings(num_tokens)
    model.save_pretrained('tmp/CustomModel/')
    return TFBertForMaskedLM.from_pretrained('tmp/CustomModel', from_pt=True)

def find_sub_list(sl,l):
    '''
    Returns the start and end positions of sublist sl in l
    '''
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1

def tokenize_and_label(annot_text, tokenizer):
    '''
    Input: a dictionary of annotated segments and main text
    Output: a dictionary of encoded text and labels
    Labels: start_claim=1, cont_claim=2, start_premise=3, cont_premise=4, none=0
    '''
    cleaned_text = clean_pipeline(annot_text['text'], False)
    encoded_text = tokenizer(cleaned_text, truncation=True, return_tensors='tf')  
    enc_tokens = list(encoded_text['input_ids'].numpy()[0])
    labels = [0 for _ in enc_tokens]
    for claim in annot_text['claim']:
        claim = clean_pipeline(claim, False)
        indices = find_sub_list(tokenizer.encode(claim)[1:-1], enc_tokens)
        if indices is not None:
            labels[indices[0]] = 1
            for i in range(indices[0]+1, indices[1]+1):
                labels[i] = 2
    for premise in annot_text['premise']:
        premise = clean_pipeline(premise, False)
        indices = find_sub_list(tokenizer.encode(premise)[1:-1], enc_tokens)
        if indices is not None:
            labels[indices[0]] = 3
            for i in range(indices[0]+1, indices[1]+1):
                labels[i] = 4
    return encoded_text, [labels]

def train_test_split():
    with open('../Data/annotated_threads.json', 'r') as f:
        data = json.load(f)
    train_keys, test_keys = list(data.keys())[:20], list(data.keys())[20:]
    return train_keys, test_keys

def make_pretrain_batch(lines, batch_size):
    batch = []
    _batch_size = 0
    for l in sorted(lines, key=lambda x:len(x.split())):
      batch.append(l)
      _batch_size += len(l.split())
      if _batch_size>=batch_size:
        yield batch
        batch = []
        _batch_size = 0
