import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


embedding_dim = 100
max_length = 16
trunc_type='post'
padding_type='post'


#args = {"sentence":"Ive had a long day"}

def predict_sentiment(args):
  sent = args["sentence"]
  sent = np.array([sent])
  
  #Load the previously built Tokenizer
  with open('models/sentiment140_tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

  #Pad sequences
  sequences = loaded_tokenizer.texts_to_sequences(sent)
  padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
  test_example = padded
  
  #Load Model
  model = tf.keras.models.load_model("models/model_conv1D_LSTM_with_batch_100_epochs.h5")
  pred_conf = model.predict(test_example)
  pred_class = (model.predict(test_example) > 0.5).astype("int32")
  
  #print("%s sentiment; %f%% confidence" % (pred_class[0][0], pred_conf[0][np.argmax(pred_conf)] * 100))
  if pred_class[0][0]==0:
    sentiment = 'Negative'
    conf = 100 - pred_conf[0][np.argmax(pred_conf)] * 100
  else:
    sentiment = 'Positive'
    conf = pred_conf[0][np.argmax(pred_conf)] * 100
  #return {"sent": sentiment, "confidence": str(pred_conf[0][np.argmax(pred_conf)] * 100)+' %'}
  return {"sentiment": sentiment, "confidence": round(conf, 3) }
  