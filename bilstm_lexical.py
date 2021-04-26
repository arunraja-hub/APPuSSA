

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import io
import time
import six
from six.moves import range



PUNCTUATION_MARKS = [
  '<FULL_STOP>', '<COMMA>', '<QUESTION_MARK>', '<EXCLAMATION_MARK>', '<DOTS>'
]

def preprocess_sentence(sentence):
  output_words = []
  output_punctuation_marks = []

  id_ = sentence.split()[0]
  words = sentence.split()[1:]
  for (word, punctuation_mark) in zip(words, words[1:] + [None]):
    if word in PUNCTUATION_MARKS:
      continue

    if punctuation_mark not in PUNCTUATION_MARKS:
      punctuation_mark = "<SPACE>"

    output_words.append(word)
    output_punctuation_marks.append(punctuation_mark)

  return [id_,
      " ".join(output_words),
      " ".join(output_punctuation_marks)
  ]


  # return [
  #     "<start> %s <end>" % " ".join(output_words),
  #     "<start> %s <end>" % " ".join(output_punctuation_marks)
  # ]

def create_dataset(path):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  word_pairs = [preprocess_sentence(l) for l in lines]
  return list(zip(*word_pairs))

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)
  # fit_on_texts
  print('lang_tokenizer.word_index',lang_tokenizer.word_index)
  print('lang_tokenizer.index_word',lang_tokenizer.index_word)


  tensor = lang_tokenizer.texts_to_sequences(lang)

  return tensor, lang_tokenizer

def load_dataset(path):
    id_,w,p = create_dataset(path)
    inp, targ = list(w),list(p)
    # input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    # target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return inp,targ
    # input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer



train_inp, train_targ = load_dataset('train.txt')
dev_inp, dev_targ = load_dataset('dev.txt')
total_inp = train_inp + dev_inp
total_targ = train_targ + dev_targ

inp_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
inp_tokenizer.fit_on_texts(total_inp)

targ_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
targ_tokenizer.fit_on_texts(total_targ)


train_input_tensor = inp_tokenizer.texts_to_sequences(train_inp)
train_target_tensor = targ_tokenizer.texts_to_sequences(train_targ)


dev_input_tensor = inp_tokenizer.texts_to_sequences(dev_inp)
dev_target_tensor = targ_tokenizer.texts_to_sequences(dev_targ)


print(inp_tokenizer.index_word)
print(targ_tokenizer.index_word)

# train_input_tensor, train_target_tensor, train_inp_lang, train_targ_lang = load_dataset('train.txt')
# dev_input_tensor, dev_target_tensor, dev_inp_lang, dev_targ_lang = load_dataset('dev.txt')

print(len(train_input_tensor))
print(len(dev_input_tensor))
print(len(list(inp_tokenizer.word_index.keys())))
print(len(list(targ_tokenizer.word_index.keys())))

def convert(lang, tensor):
  for t in tensor:
    # print(t)
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[int(t)]))

print ("Input Language; index to word mapping")
# convert(dev_inp_lang, dev_input_tensor[143])
convert(inp_tokenizer, train_input_tensor[143])
# print(convert(train_inp_lang, dev_input_tensor[143]))
print ()
print ("Target Language; index to word mapping")
convert(targ_tokenizer, train_target_tensor[143])

def bucketer(input_ds,target_ds,BATCH_SIZE):
    

    BUFFER_SIZE = 128
    # BATCH_SIZE = 64

    
    lengths = []
    for i in range(len(target_ds)):
        lengths.append(len(target_ds[i]))
    print(max(lengths))
        
    
    # buckets = list(range(93, 1309, 50))
    # min(lengths)
    buckets = list(range(5, max(lengths)+6, 5))
    print(buckets)
    batch_sizes = [BATCH_SIZE] * (len(buckets)+1)
    print(batch_sizes)
    
    
    def generator():
      for i in range(len(input_ds)):
        yield np.array(input_ds[i]),(target_ds[i])
    
    # def element_length_fn(x,y,z):
    #     return tf.shape(x)[0]
    # # ,tf.shape(y)[0],tf.shape(z)[0]
    
    dataset = tf.data.Dataset.from_generator(generator, (tf.int64, tf.int64), ([None], [None]))
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
    lambda input_ds, target_ds: tf.size(input_ds), 
    buckets,
    batch_sizes,
    drop_remainder=True,
    pad_to_bucket_boundary=False))
    
    c=0
    for i in dataset:
    #     print(ds[i])
        print(tf.shape(i[0]),  tf.shape(i[1]))
        c+=1
    
        if c == 5:
            break
    return dataset

BATCH_SIZE = 64
train_dataset = bucketer(train_input_tensor,train_target_tensor,BATCH_SIZE)
val_dataset = bucketer(dev_input_tensor,dev_target_tensor,BATCH_SIZE)

embedding_dim = 256
print(train_dataset.reduce(0, lambda x, _: x + 1).numpy())
print(val_dataset.reduce(0, lambda x, _: x + 1).numpy())

import tensorflow as tf
import sys

from tensorflow.keras.metrics import *

class StatefulF1(Metric):
  def __init__(self, name='stateful_f1', **kwargs):
    # initializing an object of the super class
    super(StatefulF1, self).__init__(name=name, **kwargs)

    # initializing state variables
    self.c = self.add_weight(name='c',initializer='zeros',dtype='int32')
    self.s = self.add_weight(name='s',initializer='zeros',dtype='int32') 
    self.d = self.add_weight(name='d',  initializer='zeros',dtype='int32') 
    self.i = self.add_weight(name='i',  initializer='zeros',dtype='int32') 


  def update_state(self, ytrue, ypred, sample_weight=None):
    # casting ytrue and ypred as float dtype
    ytrue = tf.cast(ytrue, tf.int32)
    ypred = tf.argmax(ypred,axis=2)
    ypred = tf.cast(ypred, tf.int32)





    # targ_labels = [targ_lang.word_index['<full_stop>'],targ_lang.word_index['<comma>'],
    #                targ_lang.word_index['<question_mark>'],targ_lang.word_index['<exclamation_mark>'],
    #                targ_lang.word_index['<dots>']]

    

    match = tf.equal(ytrue,ypred)
    space = tf.cast(targ_tokenizer.word_index['<space>'],tf.int32)
    pad = tf.cast(0,tf.int32)
    
    targ_space = tf.equal(space,ytrue) 
    targ_pad = tf.equal(pad,ytrue)
  
    pred_space = tf.equal(space,ypred) 
    pred_pad =tf.equal(pad,ypred)
    

    #CSID
    # Correct if targ matches pred  and targ is not a spece or pad
    self.c.assign_add(tf.reduce_sum(tf.cast(match &  ~targ_space & ~targ_pad, tf.int32)))

    # Substitution if targ does not match pred and targ is not a pad and space and pred is not a pad and space
    self.s.assign_add(tf.reduce_sum(tf.cast(( ~match & ~targ_space & ~targ_pad & ~pred_space & ~pred_pad), tf.int32)))  

    #Insertion if target is a space or pad and pred is not a space and not a pad
    self.i.assign_add(tf.reduce_sum(tf.cast(( targ_space & ~(pred_space | pred_pad) ), tf.int32)))
    # self.i.assign_add(tf.reduce_sum(tf.cast(( (targ_space | targ_pad) & ~pred_space & ~pred_pad), tf.int32)))


    #Deletion if target is not a space and not a pad and pred is a space or pad
    self.d.assign_add(tf.reduce_sum(tf.cast(( ~targ_space & ~targ_pad & (pred_space | pred_pad)), tf.int32)))
    


  def result(self):
    self.precision = self.c / (self.c+self.s+self.i)
    self.recall = self.c / (self.c+self.s+self.d) # calculates recall

    self.micro = (2*self.precision*self.recall) / (self.precision+self.recall)

    return self.micro

  def reset_states(self):
    self.c.assign(0)
    self.s.assign(0)
    self.d.assign(0)
    self.i.assign(0)

class StatefulF1Class(Metric):
  def __init__(self, name,index, **kwargs):
    # initializing an object of the super class
    super(StatefulF1Class, self).__init__(name=name, **kwargs)

    # initializing state variables
    self.c = self.add_weight(name='c',initializer='zeros',dtype='int32')
    self.p = self.add_weight(name='p',initializer='zeros',dtype='int32')
    self.r = self.add_weight(name='r',initializer='zeros',dtype='int32') 
    # self.d = self.add_weight(name='d',  initializer='zeros',dtype='int32') 
    # self.i = self.add_weight(name='i',  initializer='zeros',dtype='int32') 
    self.index=index


  def update_state(self, ytrue, ypred, sample_weight=None):
    # casting ytrue and ypred as float dtype
    ytrue = tf.cast(ytrue, tf.int32)
    ypred = tf.argmax(ypred,axis=2)
    ypred = tf.cast(ypred, tf.int32)





    targ_labels = [targ_tokenizer.word_index['<full_stop>'],targ_tokenizer.word_index['<comma>'],
                   targ_tokenizer.word_index['<question_mark>'],targ_tokenizer.word_index['<exclamation_mark>'],
                   targ_tokenizer.word_index['<dots>']]

    




    punc = targ_labels[self.index]

    space = tf.cast(targ_tokenizer.word_index['<space>'],tf.int32)
    pad = tf.cast(0,tf.int32)
    
    match = tf.equal(ytrue,ypred) & tf.equal(ytrue, punc)
    not_match  = tf.equal(ytrue,punc) & ~tf.equal(ypred, punc)

    targ_space = tf.equal(space,ytrue) 
    targ_pad = tf.equal(pad,ytrue) 

    # pred_space = tf.equal(space,ypred) 
    # pred_pad =tf.equal(pad,ypred) 

    def tf_count_pred(t, val):
      elements_equal_to_value = tf.equal(t, val) & ~targ_pad 
      as_ints = tf.cast(elements_equal_to_value, tf.int32)
      count = tf.reduce_sum(as_ints)
      return count

    def tf_count_true(t, val):
      elements_equal_to_value = tf.equal(t, val)
      as_ints = tf.cast(elements_equal_to_value, tf.int32)
      count = tf.reduce_sum(as_ints)
      return count

    self.c.assign_add(tf.reduce_sum(tf.cast(match &  ~targ_space & ~targ_pad, tf.int32)))
    self.p.assign_add(tf_count_pred(ypred,punc))
    self.r.assign_add(tf_count_true(ytrue,punc))


    

  def result(self):
    self.precision = self.c / self.p
    self.recall = self.c / self.r 

    self.micro = (2*self.precision*self.recall) / (self.precision+self.recall)

    return self.micro

  def reset_states(self):
    self.c.assign(0)
    self.p.assign(0)
    self.r.assign(0)
 

stateful_f1 = StatefulF1()
# targ_labels = [targ_tokenizer.word_index['<full_stop>'],targ_tokenizer.word_index['<comma>'],
#                    targ_tokenizer.word_index['<question_mark>'],targ_tokenizer.word_index['<exclamation_mark>'],
#                    targ_tokenizer.word_index['<dots>']]
stateful_f1_fullstop = StatefulF1Class(name='stateful_f1_fullstop',index=0)
stateful_f1_comma = StatefulF1Class(name='stateful_f1_comma',index=1)
stateful_f1_question = StatefulF1Class(name='stateful_f1_question',index=2)
stateful_f1_exclamation = StatefulF1Class(name='stateful_f1_exclamation',index=3)
stateful_f1_dots = StatefulF1Class(name='stateful_f1_dots',index=4)

vocab_inp_size = len(inp_tokenizer.word_index)+1
vocab_tar_size = len(targ_tokenizer.word_index)+1
print(vocab_inp_size)
print(vocab_tar_size)

from tensorflow import keras
from tensorflow.keras.layers import *
model = keras.Sequential()
model.add(Embedding(vocab_inp_size, 512, mask_zero=True))
# Embedding(num_input_words, hidden_layer_size, mask_zero=True)
model.add(Bidirectional(LSTM(256, return_sequences=True)))
#256
# model.add(TimeDistributed(Dense(512, activation='relu')))
# model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocab_tar_size, activation='softmax'))) 

# Compile model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=[stateful_f1,stateful_f1_fullstop,stateful_f1_comma,stateful_f1_question,stateful_f1_exclamation,stateful_f1_dots])
model.fit(train_dataset,epochs=10,batch_size=64,validation_data=val_dataset)

