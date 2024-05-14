
import time
import json
from termcolor import colored
import numpy as np
import tensorflow_text as tf_text
import tensorflow as tf
import argparse
import transformer_utils 
from utils import *


def parse_args():
    parser = argparse.ArgumentParser("T5 for question and answering")
    parser.add_argument("--seed", type=int, default=1010, help="random seed value")
    parser.add_argument("--tokenizer-model-dir", type=str, default="./pretrained_models/sentencepiece.model", help="directory of the pre-trained tokenizer")
    parser.add_argument("--pretrain-data-dir", type=str, default="./data/c4-en-10k.jsonl", help="directory of the pre-trained dataset")
    parser.add_argument("--pretrain-model-dir", type=str, default="./pretrained_models/model_c4", help="directory of the pre-trained model")
    parser.add_argument("--finetune-data-dir", type=str, default="./data/train-v2.0.json", help="directory of the fine-tuned model")
    parser.add_argument("--finetune-model-dir", type=str, default="./pretrained_models/model_qa3", help="directory of the fine-tune model")
    parser.add_argument("--encoder-maxlen", type=int, default=150, help="maximum length of encoder input")
    parser.add_argument("--decoder-maxlen", type=int, default=50, help="maximum length of decoder input")
    parser.add_argument("--num-layers", type=int, default=2, help="number of encoder and decoder layers in T5")
    parser.add_argument("--embedding-dim", type=int, default=128, help="embedding dimension in T5")
    parser.add_argument("--fully-connected-dim", type=int, default=128, help="hidden dimention of the fully connected layer in T5")
    parser.add_argument("--num-heads", type=int, default=2, help="number of heads in MHA in T5")
    parser.add_argument("--positional-encoding-length", type=int, default=256, help="maximum ength of positional encoding")
    parser.add_argument("--buffer-size", type=int, default=10000, help="dataset buffer size")
    parser.add_argument("--batch-size", type=int, default=64, help="dataset batch size")
    parser.add_argument("--pretrain-epochs", type=int, default=1, help="number of epochs for pre-training")
    parser.add_argument("--finetune-epochs", type=int, default=1, help="number of epochs for fine-tuning")
    
    parser.add_argument("--pretrain", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=False)
    parser.add_argument("--test-idx", type=int, default=10408)
    return parser.parse_args()

def main(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)


    with open(args.tokenizer_model_dir, "rb") as f:
        pre_trained_tokenizer = f.read()
        
    tokenizer = tf_text.SentencepieceTokenizer(pre_trained_tokenizer, out_type=tf.int32)



    encoder_vocab_size = int(tokenizer.vocab_size())
    decoder_vocab_size = encoder_vocab_size

    transformer = transformer_utils.Transformer(
        args.num_layers, 
        args.embedding_dim, 
        args.num_heads, 
        args.fully_connected_dim,
        encoder_vocab_size, 
        decoder_vocab_size, 
        args.positional_encoding_length, 
        args.positional_encoding_length,
    )
    learning_rate = transformer_utils.CustomSchedule(args.embedding_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    if args.pretrain:
        print('start pre-training')
        with open(args.pretrain_data_dir, 'r') as file:
            example_jsons = [json.loads(line.strip()) for line in file]
        natural_language_texts = [example_json['text'] for example_json in example_jsons]
        
        inputs_targets_pairs = [tokenize_and_mask(text.encode('utf-8', errors='ignore').decode('utf-8'), tokenizer=tokenizer) 
                                for text in natural_language_texts[0:2000]]


        inputs = tf.keras.preprocessing.sequence.pad_sequences([x[0] for x in inputs_targets_pairs], maxlen=args.encoder_maxlen, padding='post', truncating='post')
        targets = tf.keras.preprocessing.sequence.pad_sequences([x[1] for x in inputs_targets_pairs], maxlen=args.decoder_maxlen, padding='post', truncating='post')

        inputs = tf.cast(inputs, dtype=tf.int32)
        targets = tf.cast(targets, dtype=tf.int32)

        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(args.buffer_size).batch(args.batch_size)

        losses = []
        for epoch in range(args.pretrain_epochs):
            
            start = time.time()
            train_loss.reset_state()
            number_of_batches=len(list(enumerate(dataset)))

            for (batch, (inp, tar)) in enumerate(dataset):

                print(f'Epoch {epoch+1}, Batch {batch+1}/{number_of_batches}', end='\r')
                transformer_utils.train_step(inp, tar, transformer, loss_object, optimizer, train_loss)
            
            print (f'Epoch {epoch+1}, Loss {train_loss.result():.4f}')
            losses.append(train_loss.result())
            
            print (f'Time taken for one epoch: {time.time() - start} sec')

        transformer.save_weights('./model_c4_temp')

    else:
        print('loading the pre-trained model')
        transformer.load_weights(args.pretrain_model_dir)

    with open(args.finetune_data_dir, 'r') as f:
        example_jsons = json.load(f)

    example_jsons = example_jsons['data']
    inputs, targets =  parse_squad(example_jsons)   
    inputs_train = inputs[0:40000] 
    targets_train = targets[0:40000]  

    if args.finetune:

        print('start fine-tuning')

        inputs_str = [tokenizer.tokenize(s) for s in inputs_train]
        targets_str = [tf.concat([tokenizer.tokenize(s), [1]], 0) for s in targets_train]

        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs_str, maxlen=args.encoder_maxlen, padding='post', truncating='post')
        targets = tf.keras.preprocessing.sequence.pad_sequences(targets_str, maxlen=args.decoder_maxlen, padding='post', truncating='post')

        inputs = tf.cast(inputs, dtype=tf.int32)
        targets = tf.cast(targets, dtype=tf.int32)

        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(args.buffer_size).batch(args.batch_size)
        
        losses = []
        for epoch in range(args.finetune_epochs):
            
            start = time.time()
            train_loss.reset_states()
            number_of_batches=len(list(enumerate(dataset)))

            for (batch, (inp, tar)) in enumerate(dataset):
                print(f'Epoch {epoch+1}, Batch {batch+1}/{number_of_batches}', end='\r')
                transformer_utils.train_step(inp, tar, transformer, loss_object, optimizer, train_loss)
            
            print (f'Epoch {epoch+1}, Loss {train_loss.result():.4f}')
            losses.append(train_loss.result())
            
            print (f'Time taken for one epoch: {time.time() - start} sec')
            if epoch % 15 == 0:
                transformer.save_weights('./pretrained_models/model_qa_temp')
        transformer.save_weights('./pretrained_models/model_qa_temp')

    else:
        print('loading the fine-tuned model')
        transformer.load_weights(args.finetune_model_dir)
        

    print(f'start evaluating the model for idx: {args.test_idx}')
    result = transformer_utils.answer_question(inputs_train[args.test_idx], transformer, tokenizer)
    print(colored(tokenizer.detokenize(result).numpy()[0], 'blue'))
    print()
    print(inputs_train[args.test_idx])
    print(colored(targets_train[args.test_idx], 'green'))



if __name__ == '__main__':
    arglist = parse_args()
    main(arglist)