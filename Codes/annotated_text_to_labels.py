'''
Process annoatated data structured as: 
	list of dictionaries:
		{main text, list of claims, list of premises}
into
	list of dictionaries:
		{int-tokenized text, labels}
Tokenized by BertTokenizer (base uncased)
'''
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
    '''
    encoded_text = tokenizer(annot_text['text'], truncation=True, return_tensors='tf')
    enc_tokens = list(encoded_text['input_ids'].numpy())
    labels = [0 for _ in enc_tokens]
    for claim in annot_text['claim']:
        start, end = find_sub_list(tokenizer.encode(claim)[1:-1], enc_tokens)
        for i in range(start, end+1):
            labels[i] = 1
    for premise in annot_text['premise']:
        start, end = find_sub_list(tokenizer.encode(premise)[1:-1], enc_tokens)
        for i in range(start, end+1):
            labels[i] = 2
    return encoded_text, labels

