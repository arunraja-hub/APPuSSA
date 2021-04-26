import h5py
import numpy as np
import io
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from numpy import *
from operator import add
import time
import math
from tensorflow.keras.metrics import *
from tensorflow.keras.layers import *
from tensorflow import keras
from tensorflow.keras import *
K = keras.backend




def convert_h5_np(h5):
    with h5py.File(h5, "r") as f:
        sample_names = list(f.keys())
        feats = []
        for i in f.keys():
            feats.append(np.array(f[i]))
        return sample_names,feats




def get_mean_std_from_audio_features(path):
    sum_ = np.zeros((43,))
    sum_sq = np.zeros((43,))
    n = 0
    

    # with ReadHelper(path) as reader:
    for feats in train_feats:
        # print('feats.shape',feats.shape)
        nframes, nfeats = feats.shape
        n += nframes
        sum_ += feats.sum(0)
        sum_sq += (feats*feats).sum(0)
            
    print(sum_.shape)
    print(sum_sq.shape)
    mean = np.asarray(sum_/n, dtype=float)
    std = np.asarray(np.sqrt(sum_sq/n - mean**2), dtype=float)

    return mean, std



# <utterance_id> <channel_id> <start_time> <duration> <word_spoken>
def get_time_boundaries_from_ctm_file(audio,path):
    time_boundaries = defaultdict(lambda: [])

    with open(path, 'r') as f:
        for line in f:
            uttid, _, start, duration, _ = line.strip().split()
            time_boundaries[uttid].append((int(((float(start) + float(duration)) * 100) )) )
    
    print('time_boundaries ',len(list(time_boundaries.keys())))
    time_boundaries_uttids = list(time_boundaries.keys())
    
    for uttid in time_boundaries_uttids:
        if uttid not in audio:
            del time_boundaries[uttid]

        
    return time_boundaries







def preprocess_sentence(sentence):
    PUNCTUATION_MARKS = [
  '<FULL_STOP>', '<COMMA>', '<QUESTION_MARK>', '<EXCLAMATION_MARK>', '<DOTS>']
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
      "<start> %s <end>" % " ".join(output_words),
      "<start> %s <end>" % " ".join(output_punctuation_marks)]

def create_dataset(path):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [preprocess_sentence(l) for l in lines ]
    return list(zip(*word_pairs))

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)

    return tensor, lang_tokenizer




def load_dataset(path):
    id_,w,p = create_dataset(path)
    inp, targ = list(w),list(p)
    return id_,inp,targ





def subsample_with_cnn(X, subsampling_factor):
  for i in range(int(math.log(subsampling_factor,2))):
    X = tf.keras.layers.Conv1D(filters=128, kernel_size=2, strides=2,padding="causal", activation='relu')(X)
    X = tf.keras.layers.BatchNormalization()(X)
  return X

class HierarchicalEncoder(tf.keras.Model):
    def __init__(self,enc_units, batch_size):
        super(HierarchicalEncoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.enc_units, return_sequences=True))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.enc_units,return_sequences=True,return_state=True))
        self.ln1 = tf.keras.layers.LayerNormalization(axis= -1)
        self.ln2 = tf.keras.layers.LayerNormalization(axis= -1)
        self.bn1 = tf.keras.layers.BatchNormalization(axis= -1)
        self.bn2 = tf.keras.layers.BatchNormalization(axis= -1)
        


    def call(self, frame,hidden,bounds):
        
        subsampling_factor = 8
        
        
        self.bounds = bounds  // subsampling_factor
       
        
        frame = self.bn1(frame)
        frame = subsample_with_cnn(frame, subsampling_factor) 
        frame = self.bn2(frame)
        
        
        lstm_output= tf.convert_to_tensor(self.lstm(frame))
        lstm_output = self.ln1(lstm_output)
        

        self.bounds = tf.cast(self.bounds,dtype=tf.int32)
        ind = self.bounds

        rn = tf.range(tf.shape(ind)[0]) #batch size 64
        


        rn_rep = tf.repeat(rn,tf.shape(ind)[1])#repeat rn shape[1] times
        
        


        rn_rep=tf.convert_to_tensor(tf.split(rn_rep,self.batch_size)) 

        bounds_ = tf.stack([rn_rep,ind], axis=2)   
    
        word_acoustic_rep = tf.gather_nd(lstm_output, bounds_,batch_dims=0)
        

        
        whole_sequence_output, h_forward, c_forward, h_backward, c_backward = self.lstm2(word_acoustic_rep)
        
        whole_sequence_output = self.ln2(whole_sequence_output)
        h = tf.concat([h_forward,h_backward],1)
        c = tf.concat([c_forward,c_backward],1)

        
        return whole_sequence_output, [h,c]
    
    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_size, self.enc_units)) for _ in range(2)]









class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query = tf.concat(query, axis=-1)
        query_with_time_axis = tf.expand_dims(query, 1)
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)

        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights








def bucketer(input_ds,target_ds,BATCH_SIZE):
    

    BUFFER_SIZE = 128
    # BATCH_SIZE = 64
    
    
    lengths_audio = []
    for i in input_ds:
        lengths_audio.append(len(input_ds[i][0]))
    print('max length of acosutic seq:',max(lengths_audio),min(lengths_audio))

    
    lengths = []
    for i in input_ds:
        lengths.append(len(target_ds[i]))
    print('max length of punctuation seq:',max(lengths),min(lengths))
        
    
    # buckets = list(range(93, 1309, 50))
    # min(lengths)
    buckets = list(range(min(lengths), max(lengths)+1, 5))
    print(buckets)
    batch_sizes = [BATCH_SIZE] * (len(buckets)+1)
    print(batch_sizes)
    
    
    
    print('generator')
    # train_input[i] = [train_id_word_dict[i],train_audio[i],tf.convert_to_tensor(train_boundaries[i])]
    def generator():
        for i in input_ds:
            yield (np.array(input_ds[i][0])[1:-1],np.array(input_ds[i][1], dtype = np.float32),np.array(input_ds[i][2]),np.array(target_ds[i])[:-2]), np.array(target_ds[i])[1:-1]

    print('element_length_fn')

    def element_length_fn(x,y):
        return tf.shape(y)
    # tf.shape(x[0])[0]
    # ,tf.shape(x[1])[0],
    
    print('dataset')
    dataset = tf.data.Dataset.from_generator(generator, ((tf.int64,tf.float32, tf.int64,tf.int64),tf.int64), (([None],[None,43],[None], [None]),[None]))
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
    element_length_fn, 
    buckets,
    batch_sizes,
    drop_remainder=True,
    pad_to_bucket_boundary=False))
    
    c=0
    for i in dataset:
    #     print(ds[i])
        # print(i)
        print(i[0][0].shape,i[0][1].shape,i[0][2].shape,i[0][3].shape,i[1].shape)
        c+=1
    
        if c == 5:
            break
        
    return dataset


    




class Decoder(Layer):

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = Embedding(vocab_tar_size, 256, mask_zero=True)
        self.decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
        self.decoder_dense = Dense(vocab_tar_size, activation='softmax')
        self.attn = BahdanauAttention(256)
        self.concat = tf.keras.layers.Concatenate(axis=0, name='concat_layer')
       
        

    @tf.function
    def decoder_loop(self,decoder_inputs,states):
           
          
          decoder_inputs = tf.cast(decoder_inputs,dtype=tf.int64)

          inputs = K.in_train_phase(decoder_inputs,states[2])
          inputs = self.embedding(inputs)

          
          # attn_out
          context_vector, attention_weights= self.attn(states[0], self.encoder_outputs)
          
          inputs = tf.concat([tf.expand_dims(context_vector, 1), inputs], axis=-1)
          
          outputs, h,c = self.decoder_lstm(inputs,states[:2])

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
        
        





# In[3]:

if __name__ == '__main__':
    

    # EPOCHS = 50
    BATCH_SIZE = 64


    

    
   
    print('start')
    train_sample,train_feats = convert_h5_np("train_feats")
    dev_sample,dev_feats = convert_h5_np("dev_feats")
    print('feats generated, sample lengths are')
    print(len(train_sample))
    print(len(dev_sample))
    
    train_audio = dict(zip(train_sample,train_feats ))
    print('len train_audio',len(list(train_audio.keys())))
    dev_audio = dict(zip(dev_sample,dev_feats ))
    print('len dev_audio',len(list(dev_audio.keys())))
    
    
    
    train_ids,train_inp, train_targ = load_dataset('train.txt')
    dev_ids, dev_inp, dev_targ = load_dataset('dev.txt')
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
    
    
    
    
    
    train_boundaries = get_time_boundaries_from_ctm_file(train_audio,'train.ctm')
    # train_ids,_, train_target_tensor, _, train_targ_lang = load_dataset('train.txt')
    train_id_word_dict = dict(zip(train_ids, train_input_tensor))
    train_id_punc_dict = dict(zip(train_ids, train_target_tensor))
    print('train bounds done')
    #
    dev_boundaries = get_time_boundaries_from_ctm_file(dev_audio,'dev.ctm')
    # dev_ids,_, dev_target_tensor, _, dev_targ_lang = load_dataset('dev.txt')
    dev_id_word_dict = dict(zip(dev_ids, dev_input_tensor))
    dev_id_punc_dict = dict(zip(dev_ids, dev_target_tensor))
    print('dev bounds done')

    train_input = {}
    train_target = {}
    #
    # for i in train_audio:
    #     print(i)
    #     break
    #
    for i in train_audio:
        train_input[i] = [train_id_word_dict[i],train_audio[i],tf.convert_to_tensor(train_boundaries[i])]
        train_target[i] = tf.convert_to_tensor(train_id_punc_dict[i])
    print('train ds done')
        
    val_input = {}
    val_target = {}
    for i in dev_audio:
        val_input[i] = [dev_id_word_dict[i],dev_audio[i],tf.convert_to_tensor(dev_boundaries[i])]
        val_target[i] = tf.convert_to_tensor(dev_id_punc_dict[i])
    print('val ds done')
        
    train_dataset = bucketer(train_input,train_target,BATCH_SIZE)
    val_dataset = bucketer(val_input,val_target,BATCH_SIZE)
    

    
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
          
    
    
      def result(self):
        self.precision = self.c / self.p
        self.recall = self.c / self.r 
    
        self.micro = (2*self.precision*self.recall) / (self.precision+self.recall)
    
        return self.micro
    
      def reset_states(self):
        self.c.assign(0)
        self.p.assign(0)
        self.r.assign(0)


    print('setting up metrics')
    stateful_f1 = StatefulF1()
    stateful_f1_fullstop = StatefulF1Class(name='stateful_f1_fullstop',index=0)
    stateful_f1_comma = StatefulF1Class(name='stateful_f1_comma',index=1)
    stateful_f1_question = StatefulF1Class(name='stateful_f1_question',index=2)
    stateful_f1_exclamation = StatefulF1Class(name='stateful_f1_exclamation',index=3)
    stateful_f1_dots = StatefulF1Class(name='stateful_f1_dots',index=4)
    
    
    print('input and tar vocab sizes are:')
    vocab_inp_size = len(inp_tokenizer.word_index)+1
    vocab_tar_size = len(targ_tokenizer.word_index)+1
    print(vocab_inp_size)
    print(vocab_tar_size)
    # units = 256
    
    
    
    
    
    
    
##########################LEXICAL 
    
    enc_input = Input(shape=(None,))
    encoder_inputs = Embedding(vocab_inp_size, 512, mask_zero=True)(enc_input)
    encoder_lexical = Bidirectional(LSTM(128, return_sequences=True,return_state=True))
    encoder_outputs, h_forward, c_forward, h_backward, c_backward = encoder_lexical(encoder_inputs)
    
    lexical_h = tf.concat([h_forward,h_backward],1)
    lexical_c = tf.concat([c_forward,c_backward],1)
    
    # encoder_states = h + c
    # [h,c]
    
    
##########################ACOUSTIC


    encoder = HierarchicalEncoder(enc_units=128, batch_size=BATCH_SIZE)
    decoder = Decoder()
    
    enc_hidden = encoder.initialize_hidden_state()
    inp = Input(shape=(None,43))
    bounds = Input(shape=(None,))
    enc_output, enc_hidden = encoder(inp, enc_hidden,bounds)
    
    acoustic_h = enc_hidden[0]
    acoustic_c = enc_hidden[1]
    
    
##########################COMBINE

    
    combined_enc_output =   tf.convert_to_tensor(encoder_outputs) + enc_output
    combined_enc_state =  [lexical_h + acoustic_h , lexical_c + acoustic_c] 

    
    
    
    dec_input = Input(shape=(None,))
    decoder_outputs = decoder(dec_input,combined_enc_state,combined_enc_output)
    
    model = Model([enc_input, inp, bounds, dec_input], decoder_outputs)
    

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=[stateful_f1,stateful_f1_fullstop,stateful_f1_comma,stateful_f1_question,stateful_f1_exclamation,stateful_f1_dots])
    

    model.fit(train_dataset,epochs=20,batch_size=BATCH_SIZE,validation_data=val_dataset)
    
    


    
    
    
    
    