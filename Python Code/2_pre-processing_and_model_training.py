import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import pickle
import h5py

!mkdir ~/temp_data/models/
!mkdir ~/temp_data/embeddings/


embedding_dim = 100
max_length = 16
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size=1600000
test_portion=.1
num_epochs = 30    ####Best results were obtained with 100 epochs
batch_size = 16000

def build_corpus():
  num_sentences = 0
  corpus = []
  with open("temp_data/sentiment140_unzipped/clean_data.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
      text_and_label=[]
      text_and_label.append(row[5])
      label_only=row[0]
      if label_only=='0':
        text_and_label.append(0)
      else:
        text_and_label.append(1)
      num_sentences = num_sentences + 1
      corpus.append(text_and_label)
      
  print("Total sentences: " + str(num_sentences))
  print("Corpus Size: " + str(len(corpus)))
  print("Example entry from Corpus: " + str(corpus[20]))
  
  all_labels = [item[1] for item in corpus]
  unique_labels = set(all_labels)
  print("Unique labels: " + str(unique_labels))
  
  return corpus

def tokenize_and_pad():
  corpus = build_corpus()
  all_sentences=[]
  all_labels=[]
  random.shuffle(corpus)
  for x in range(training_size):
    all_sentences.append(corpus[x][0])
    all_labels.append(corpus[x][1])


  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(all_sentences)

  word_index = tokenizer.word_index
  vocab_size=len(word_index)

  sent_sequences = tokenizer.texts_to_sequences(all_sentences)
  padded_sent = pad_sequences(sent_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

  split = int(test_portion * training_size)

  test_sequences = padded_sent[0:split]
  training_sequences = padded_sent[split:training_size]
  test_labels = all_labels[0:split]
  training_labels = all_labels[split:training_size]
  
  
  print("Vocabulary Size: " + str(vocab_size))
  print("Training data shape: " + str(training_sequences.shape))
  print("Training Labels Count: " + str(len(training_labels)))
  print("Validation data shape: " + str(test_sequences.shape))
  print("Validation Labels Count: " + str(len(test_labels)))

# saving the Tokenizer
  with open('temp_data/models/test_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
  return training_sequences, training_labels, test_sequences, test_labels, word_index, vocab_size

def embed_data(vocab_size, word_index):
  url = 'http://nlp.stanford.edu/data/glove.6B.zip'
  with urlopen(url) as response:
    with ZipFile(BytesIO(response.read())) as zfile:
      zfile.extractall('temp_data/embeddings/glove_6B')

  embedding_dict = {};
  with open('temp_data/embeddings/glove_6B/glove.6B.100d.txt') as f:
    for item in f:
      embed_items = item.split();
      word = embed_items[0];
      word_vec = np.asarray(embed_items[1:], dtype='float32');
      embedding_dict[word] = word_vec;

  embedding_matrix = np.zeros((vocab_size+1, embedding_dim));
  for word, i in word_index.items():
    embedding_vector = embedding_dict.get(word);
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector;
  
  return embedding_dict, embedding_matrix


def build_model(vocab_size, embedding_matrix):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False),
    tf.keras.layers.Conv1D(32, 8, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation = 'relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
  model.summary()
  return model


def train_model(vocab_size, embedding_matrix, training_sequences, training_labels, batch_size, num_epochs, test_sequences, test_labels):
  model = build_model(vocab_size, embedding_matrix)  
  history = model.fit(np.array(training_sequences), np.array(training_labels), batch_size=batch_size, epochs=num_epochs, validation_data=(np.array(test_sequences), np.array(test_labels)), verbose=2)
  
  ##Save the model
  model.save('temp_data/models/test_model.h5')
  return model, history

def plot_model(vocab_size, embedding_matrix, training_sequences, training_labels, batch_size, num_epochs, test_sequences, test_labels):
  model_and_history = train_model(vocab_size, embedding_matrix, training_sequences, training_labels, batch_size, num_epochs, test_sequences, test_labels)
  model = model_and_history[0]
  history = model_and_history[1]
  print("Training Complete!")
  
  #-----------------------------------------------------------
  # Retrieve a list of list results on training and test data
  # sets for each training epoch
  #-----------------------------------------------------------
  acc=history.history['acc']
  val_acc=history.history['val_acc']
  loss=history.history['loss']
  val_loss=history.history['val_loss']

  epochs=range(len(acc)) # Get number of epochs
  
  #------------------------------------------------
  # Plot training and validation accuracy per epoch
  #------------------------------------------------
  plt.plot(epochs, acc, 'r')
  plt.plot(epochs, val_acc, 'b')
  plt.title('Training and validation accuracy')
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend(["Accuracy", "Validation Accuracy"])

  plt.figure()
  
  #------------------------------------------------
  # Plot training and validation loss per epoch
  #------------------------------------------------
  plt.plot(epochs, loss, 'r')
  plt.plot(epochs, val_loss, 'b')
  plt.title('Training and validation loss')
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend(["Loss", "Validation Loss"])

  plt.figure()
  return model

#Function Calls

final_data = tokenize_and_pad()
training_sequences = final_data[0]
training_labels = final_data[1]
test_sequences = final_data[2]
test_labels = final_data[3]
word_index = final_data[4]
vocab_size = final_data[5]

embedding_data = embed_data(vocab_size, word_index)
embedding_dict = embedding_data[0]
embedding_matrix = embedding_data[1]

print("Embedding Matrix Shape: " + str(embedding_matrix.shape))

model = plot_model(vocab_size, embedding_matrix, training_sequences, training_labels, batch_size, num_epochs, test_sequences, test_labels)

# saving model
model.save('models/model_conv1D_LSTM_with_batch_100_epochs.h5')

# saving tokenizer
!cp /home/cdsw/temp_data/models/test_tokenizer.pickle /home/cdsw/models/sentiment140_tokenizer.pickle

