import pandas as pd
import numpy as np
import math
import re

from bs4 import BeautifulSoup
import random


import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import layers
import bert
from bert import tokenization
cols  = ["sentiment" , "id" , "date" , "query", "user" , "text"]
data = pd.read_csv(
    "training.txt",
    names = cols,
    header=None,
    encoding="latin1",
    )

#Clean data


data.drop(["id", "date", "query" , "user"], axis=1, inplace=True)


def clean_tweet (tweet):
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    tweet = re.sub(r"https?://[A-Za-z0-9/.]+", ' ', tweet)
    tweet = re.sub(r"[^A-Za-z.!?']+", ' ', tweet)
    tweet = re.sub(r" +", ' ', tweet)
    
    return tweet
    
data_clean = [clean_tweet(tweet) for tweet in data.text ]

data_labels = data.sentiment.values
data_labels [ data_labels==4] = 1


# Tokenizer

FullTokenizer = bert.tokenization.FullTokenizer
print ("Loaded Full tokenization")
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_a-12/3", trainable = False);

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = FullTokenizer(vocab_file, do_lower_case)

import sys
from absl import flags
sys.argv=['preserve_unused_tokens=False']
flags.FLAGS(sys.argv)


def encode_sentence (sent): 
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))
    
print ("Encoding sentences")
data_inputs = [encode_sentence(sentence) for sentence in data_clean]
print ("Sentence encoded")


data_with_len = [[sent, data_labels[i], len(sent)] for i, sent in enumerate(data_inputs)]

random.shuffle (data_with_len)
data_with_len.sort(key=lambda x: x[2])
sorted_all = [(sent_lab[0], sent_lab[1]) for sent_lab in data_with_len if sent_lab[2] > 7]


all_dataset = tf.data.Dataset.from_generator(lambda: sorted_all, output_types=(tf.int32, tf.int32))

BATCH_SIZE = 32
all_batched = all_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None,),())) 

print (next(iter(all_batched)))


NB_BATCHES = math.ceil(len(sorted_all)/BATCH_SIZE)
NB_BATCHES_TEST = NB_BATCHES//10
all_batched.shuffle(NB_BATCHES)
test_dataset = all_batched.take(NB_BATCHES_TEST)
train_dataset = all_batched.skip(NB_BATCHES_TEST)

print ("Input data has been created")

class DCNN (tf.keras.Model): 
    def __init__ (self, 
                  vocab_size,
                  emb_dim = 128,
                  nb_filters = 50,
                  FFN_units = 512,
                  nb_classes = 2,
                  dropout_rate = 0.1,
                  training=False, 
                  name="dcnn"):
        super(DCNN, self).__init__(name=name)
        
        self.embedding = layers.Embedding (vocab_size,
                                          emb_dim)
        
        self.bigram = layers.Conv1D(filters=nb_filters,
                                   kernel_size=2,
                                   padding="valid",
                                   activation="relu")
                                   
        self.trigram = layers.Conv1D(filters=nb_filters,
                                   kernel_size=3,
                                   padding="valid",
                                   activation="relu")
                                   
        self.fourgram = layers.Conv1D(filters=nb_filters,
                                   kernel_size=4,
                                   padding="valid",
                                   activation="relu")
                                   


        self.pool = layers.GlobalMaxPool1D()
        
        self.dense_1 = layers.Dense (units=FFN_units, activation="relu")
        
        self.dropout = layers.Dropout (rate=dropout_rate)
        
        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
                                           
        else:
            self.last_dense = layers.Dense(units=nb_classes,
                                           activation="softmax")
                                           
                                           
    def call (self, inputs, training):
        x = self.embedding(inputs)
        
        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)        
        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)
                                           
        merged = tf.concat([x_1, x_2, x_3], axis=-1)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)
        
        return output
        
        
        
VOCAB_SIZE = len(tokenizer.vocab)
EMB_DIM = 200
NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = 2

DROPOUT_RATE = 0.2

NB_EPOCHS = 5

print ("Generating model")

Dcnn = DCNN (vocab_size = VOCAB_SIZE,
             emb_dim = EMB_DIM,
             nb_filters = NB_FILTERS,
             FFN_units = FFN_UNITS, 
             nb_classes = NB_CLASSES,
             dropout_rate = DROPOUT_RATE)
             
if NB_CLASSES ==2:
    Dcnn.compile(loss="binary_crossentropy",
                 optimizer="adam",
                 metrics=["accuracy"])
                 
else:
    Dcnn.compile(loss="binary_crossentropy",
                 optimizer="adam",
                 metrics=["sparse_categorical_accuracy"])


checkpoint_path="."

ckpt = tf.train.Checkpoint(Dcnn=Dcnn)

ckpt_manager=tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ("Ultimo checkpoint restaurado")
    
    
    
class MyCustomCallback(tf.keras.callbacks.Callback):

    def on_epoch_end (self, epoch, logs=None):
        ckpt_manager.save()
        print ("Checkpoint guardado en {}.".format(checkpoint_path))
        
        
Dcnn.fit(train_dataset,
         epochs=NB_EPOCHS,
         callbacks=[MyCustomCallback()])
        
        
results = Dcnn.evaluate(test_dataset)
print (results)