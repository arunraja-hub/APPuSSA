

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import sys
from tensorflow.keras.metrics import *
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

  # return [id_,
  #     " ".join(output_words),
  #     " ".join(output_punctuation_marks)
  # ]


  return [id_,
      "<start> %s <end>" % " ".join(output_words),
      "<start> %s <end>" % " ".join(output_punctuation_marks)
  ]

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
        yield (np.array(input_ds[i])[1:-1],np.array(target_ds[i])[:-2]), np.array(target_ds[i])[1:-1]
    
    # def element_length_fn(x,y,z):
    #     return tf.shape(x)[0]
    # # ,tf.shape(y)[0],tf.shape(z)[0]
    
    dataset = tf.data.Dataset.from_generator(generator, ((tf.int64, tf.int64),tf.int64), (([None], [None]),[None]))
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
    # ._iter_()

# lexical_bucketer

BATCH_SIZE = 64
train_dataset = bucketer(train_input_tensor,train_target_tensor,BATCH_SIZE)

val_dataset = bucketer(dev_input_tensor,dev_target_tensor,BATCH_SIZE)


embedding_dim = 256
# print(train_dataset.reduce(0, lambda x, _: x + 1).numpy())
# print(val_dataset.reduce(0, lambda x, _: x + 1).numpy())



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
    start = tf.cast(targ_tokenizer.word_index['<start>'],tf.int32)
    end = tf.cast(targ_tokenizer.word_index['<end>'],tf.int32)
    
    targ_space = tf.equal(space,ytrue) 
    targ_pad = tf.equal(pad,ytrue)
    targ_start = tf.equal(start,ytrue) 
    targ_end = tf.equal(end,ytrue)  
  
    pred_space = tf.equal(space,ypred) 
    pred_pad =tf.equal(pad,ypred)
    pred_start = tf.equal(start,ypred) 
    pred_end =tf.equal(end,ypred)
    

    #CSID
    # Correct if targ matches pred  and targ is not a spece or pad
    self.c.assign_add(tf.reduce_sum(tf.cast(match &  ~targ_space & ~targ_pad & ~targ_end & ~targ_start, tf.int32)))

    # Substitution if targ does not match pred and targ is not a pad and space and pred is not a pad and space
    self.s.assign_add(tf.reduce_sum(tf.cast(( ~match & ~targ_space & ~targ_pad & ~targ_end & ~targ_start & ~pred_space & ~pred_pad & ~pred_end & ~pred_start), tf.int32)))  

    #Insertion if target is a space or pad and pred is not a space and not a pad
    self.i.assign_add(tf.reduce_sum(tf.cast(( targ_space & ~(pred_space | pred_pad | pred_start | pred_end) ), tf.int32)))
    # self.i.assign_add(tf.reduce_sum(tf.cast(( (targ_space | targ_pad) & ~pred_space & ~pred_pad), tf.int32)))


    #Deletion if target is not a space and not a pad and pred is a space or pad
    self.d.assign_add(tf.reduce_sum(tf.cast(( ~targ_space & ~targ_pad & (pred_space | pred_pad | pred_start | pred_end)), tf.int32)))
    


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
    start = tf.cast(targ_tokenizer.word_index['<start>'],tf.int32)
    end = tf.cast(targ_tokenizer.word_index['<end>'],tf.int32)

    
    match = tf.equal(ytrue,ypred) & tf.equal(ytrue, punc)
    not_match  = tf.equal(ytrue,punc) & ~tf.equal(ypred, punc)

    targ_space = tf.equal(space,ytrue) 
    targ_pad = tf.equal(pad,ytrue)
    targ_start = tf.equal(start,ytrue) 
    targ_end = tf.equal(end,ytrue)  

    # pred_space = tf.equal(space,ypred) 
    # pred_pad =tf.equal(pad,ypred) 

    def tf_count_pred(t, val):
      elements_equal_to_value = tf.equal(t, val) & ~targ_pad & ~targ_space & ~targ_end & ~targ_start
      as_ints = tf.cast(elements_equal_to_value, tf.int32)
      count = tf.reduce_sum(as_ints)
      return count

    def tf_count_true(t, val):
      elements_equal_to_value = tf.equal(t, val)
      as_ints = tf.cast(elements_equal_to_value, tf.int32)
      count = tf.reduce_sum(as_ints)
      return count

    self.c.assign_add(tf.reduce_sum(tf.cast(match & ~targ_pad & ~targ_space & ~targ_end & ~targ_start, tf.int32)))
    self.p.assign_add(tf_count_pred(ypred,punc))
    self.r.assign_add(tf_count_true(ytrue,punc))


    

    # self.c.assign_add(tf.reduce_sum(tf.cast(match &  ~targ_space & ~targ_pad, tf.int32)))
    # self.s.assign_add(tf.reduce_sum(tf.cast(( not_match & ~targ_space & ~targ_pad & ~pred_space & ~pred_pad), tf.int32)))
    # self.i.assign_add(tf.reduce_sum(tf.cast(( targ_space & ~(pred_space | pred_pad) ), tf.int32)))
    # self.d.assign_add(tf.reduce_sum(tf.cast(( ~targ_space & ~targ_pad & (pred_space | pred_pad)), tf.int32)))

    
      

    # #CSID
    # # Correct if targ matches pred  and targ is not a spece or pad
    # self.c.assign_add(tf.reduce_sum(tf.cast(match &  ~targ_space & ~targ_pad, tf.int32)))

    # # Substitution if targ does not match pred and targ is not a pad and space and pred is not a pad and space
    # self.s.assign_add(tf.reduce_sum(tf.cast(( ~match & ~targ_space & ~targ_pad & ~pred_space & ~pred_pad), tf.int32)))  

    # #Insertion if target is a space or pad and pred is not a space and not a pad
    # self.i.assign_add(tf.reduce_sum(tf.cast(( targ_space & ~(pred_space | pred_pad) ), tf.int32)))
    # # self.i.assign_add(tf.reduce_sum(tf.cast(( (targ_space | targ_pad) & ~pred_space & ~pred_pad), tf.int32)))


    # #Deletion if target is not a space and not a pad and pred is a space or pad
    # self.d.assign_add(tf.reduce_sum(tf.cast(( ~targ_space & ~targ_pad & (pred_space | pred_pad)), tf.int32)))
      


  def result(self):
    self.precision = self.c / self.p
    self.recall = self.c / self.r 

    self.micro = (2*self.precision*self.recall) / (self.precision+self.recall)

    return self.micro

  def reset_states(self):
    self.c.assign(0)
    self.p.assign(0)
    self.r.assign(0)
    # self.d.assign(0)
    # self.i.assign(0)

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
units = 256

targ_tokenizer.word_index

from tensorflow.keras import models
from numpy import array_equal
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True))

  def call(self, x, hidden):
    x = self.embedding(x)
    x = tf.reshape(x,[64,9,256])
    outputs = self.gru(x,initial_state = hidden)
    return outputs[0], outputs[-1]

  def initialize_hidden_state(self):
    return [tf.zeros((self.batch_sz, self.enc_units)) for _ in range(2)]

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

from tensorflow.keras.layers import Layer
from tensorflow import keras
K = keras.backend


class Decoder(Layer):

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = Embedding(vocab_tar_size, 256, mask_zero=True)
        self.decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
        self.decoder_dense = Dense(vocab_tar_size, activation='softmax')
        self.attn = BahdanauAttention(256)
        self.concat = tf.keras.layers.Concatenate(axis=0, name='concat_layer')
       
        

    # @tf.function
    def decoder_loop(self,decoder_inputs,states):
          
          decoder_inputs = tf.cast(decoder_inputs,dtype=tf.int64)
          # print('in training phase',decoder_inputs,states[2])

          inputs = K.in_train_phase(decoder_inputs,states[2])
          inputs = self.embedding(inputs)
          # outputs, h, c = self.decoder_lstm(inputs, initial_state=states[:2])

          
          # attn_out
          context_vector, attention_weights= self.attn(states[0], self.encoder_outputs)
          
          inputs = tf.concat([tf.expand_dims(context_vector, 1), inputs], axis=-1)
          
          outputs, h,c = self.decoder_lstm(inputs,states[:2])
          
          # outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
          
          # decoder_concat_input = self.concat([inputs,attn_out])
          # ,axis=-1

          # outputs, h,c= self.decoder_lstm(decoder_concat_input, initial_state=states[:2])


          outputs = self.decoder_dense(outputs)
          # decoder_concat_input
          pred = tf.argmax(outputs,axis = -1)
          states = [h, c,pred]
          return outputs,states


    @tf.function
    def call(self, inputs,enc_states, encoder_outputs,training = None):
        
        
        
      self.encoder_outputs = encoder_outputs
      inputs = tf.reshape(inputs,(tf.shape(inputs)[0],-1,1))
      # enc_states_new 
      enc_states_new = enc_states + [tf.ones((tf.shape(inputs)[0],1),dtype=tf.int64)]
      # (last_output, outputs, new_states)
      _, decoder_outputs, _= K.rnn(self.decoder_loop, inputs, enc_states_new)
      decoder_outputs = tf.squeeze(decoder_outputs, [2])
      return decoder_outputs





enc_input = Input(shape=(None,))
encoder_inputs = Embedding(vocab_inp_size, 512, mask_zero=True)(enc_input)
encoder = Bidirectional(LSTM(128, return_sequences=True,return_state=True))
encoder_outputs, h_forward, c_forward, h_backward, c_backward = encoder(encoder_inputs)
h = tf.concat([h_forward,h_backward],1)
c = tf.concat([c_forward,c_backward],1)
encoder_states = [h,c]


dec_input = Input(shape=(None,))
decoder = Decoder()
decoder_outputs = decoder(dec_input,encoder_states,encoder_outputs)



model = Model([enc_input, dec_input], decoder_outputs)



print('compile')
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=[stateful_f1,stateful_f1_fullstop,stateful_f1_comma,stateful_f1_question,stateful_f1_exclamation,stateful_f1_dots])

# model.fit(bucketer(train_input_tensor,train_target_tensor,BATCH_SIZE),epochs=10,batch_size=64,validation_data=bucketer(dev_input_tensor,dev_target_tensor,BATCH_SIZE))
print('fit')
model.fit(train_dataset,epochs=10,batch_size=64,validation_data=val_dataset)

