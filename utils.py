import string
import tensorflow as tf
import numpy as np

def get_sentinels(tokenizer, display=False):
    sentinels = {}
    vocab_size = tokenizer.vocab_size(name=None)
    for i, char in enumerate(reversed(string.ascii_letters), 1):
        decoded_text = tokenizer.detokenize([vocab_size - i]).numpy().decode("utf-8")
        
        sentinels[decoded_text] = f'<{char}>'    

    return sentinels

def pretty_decode(encoded_str_list, sentinels, tokenizer):
    if tf.is_tensor(encoded_str_list) and encoded_str_list.dtype == tf.string:
        for token, char in sentinels.items():
            encoded_str_list = tf.strings.regex_replace(encoded_str_list, token, char)
        return encoded_str_list
  
    return pretty_decode(tokenizer.detokenize(encoded_str_list), sentinels, tokenizer)

def tokenize_and_mask(text, 
                      noise=0.15, 
                      randomizer=np.random.uniform, 
                      tokenizer=None):
    """Tokenizes and masks a given input.

    Args:
        text (str or bytes): Text input.
        noise (float, optional): Probability of masking a token. Defaults to 0.15.
        randomizer (function, optional): Function that generates random values. Defaults to np.random.uniform.
        tokenizer (function, optional): Tokenizer function. Defaults to tokenize.

    Returns:
        inps, targs: Lists of integers associated to inputs and targets.
    """
    
    cur_sentinel_num = 0
    
    inps, targs = [], []

    vocab_size = int(tokenizer.vocab_size())
    

    eos = tokenizer.string_to_id("</s>").numpy()
    
    prev_no_mask = True
    
    for token in tokenizer.tokenize(text).numpy():
        
        rnd_val = randomizer() 
        
        if rnd_val < noise:
            
            if prev_no_mask:
                
                cur_sentinel_num += 1
                
                end_id = vocab_size - cur_sentinel_num
                
                targs.append(end_id)
                
                inps.append(end_id)
                
            targs.append(token)
            
            prev_no_mask = False

        else:
            
            inps.append(token)
            
            prev_no_mask = True
    
    
    targs.append(eos)
    
    return inps, targs

def parse_squad(dataset):
    """Extract all the answers/questions pairs from the SQuAD dataset

    Args:
        dataset (dict): The imported JSON dataset

    Returns:
        inputs, targets: Two lists containing the inputs and the targets for the QA model
    """

    inputs, targets = [], []


    for article in dataset:
        
        for paragraph in article['paragraphs']:
            
            context = paragraph['context']
            
            for qa in paragraph['qas']:
                
                if len(qa['answers']) > 0 and not(qa['is_impossible']):
                    
                    question_context = 'question: ' + qa['question'] + ' context: ' + context
                    
                    answer = 'answer: ' + qa['answers'][0]['text']
                    
                    inputs.append(question_context)
                    
                    targets.append(answer)
    
    
    return inputs, targets


