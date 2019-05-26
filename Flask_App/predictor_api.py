"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.

This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import pickle
import tensorflow as tf
from random import randint
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename,'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def seed_text_generator(seed_corpus):
    seed_text = seed_corpus[randint(0,len(seed_corpus))]
    return seed_text

# generate a sequence from a language model
def generate_quran(seed_text):
    # load the model
    model = tf.keras.models.load_model('/Users/tcbon/Desktop/Coding/Metis/Bootcamp/Project_4/Flask_App/static/models/quran_model_3.hdf5')
    model._make_predict_function()
    
    # load the tokenizer
    tokenizer = pickle.load(open('/Users/tcbon/Desktop/Coding/Metis/Bootcamp/Project_4/Flask_App/static/models/quran_tokenizer_3.pkl', 'rb'))
            
    seq_length = 50
    n_words = 30

    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

# generate a sequence from a language model
def generate_bible(seed_text):
    # load the model
    model = tf.keras.models.load_model('/Users/tcbon/Desktop/Coding/Metis/Bootcamp/Project_4/Flask_App/static/models/new_test_model_2.hdf5')
    model._make_predict_function()
    
    # load the tokenizer
    tokenizer = pickle.load(open('/Users/tcbon/Desktop/Coding/Metis/Bootcamp/Project_4/Flask_App/static/models/new_test_tokenizer_2.pkl', 'rb'))
        
    seq_length = 50
    n_words = 30

    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

def generate_torah(seed_text):
    # load the model
    model = tf.keras.models.load_model('/Users/tcbon/Desktop/Coding/Metis/Bootcamp/Project_4/Flask_App/static/models/torah_model.hdf5')
    model._make_predict_function()
    
    # load the tokenizer
    tokenizer = pickle.load(open('/Users/tcbon/Desktop/Coding/Metis/Bootcamp/Project_4/Flask_App/static/models/torah_tokenizer.pkl', 'rb'))
        
    seq_length = 50
    n_words = 30

    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

#

# This section checks that the generative code runs properly
# To run, type "python predictor_api.py" in the terminal.
#
# The if __name__='__main__' section ensures this code only runs
# when running this file; it doesn't run when importing
if __name__ == '__main__':
    seed_text = seed_text_generator(clean_quran)
    generated = generate_seq(seed_text)
    
    # generate new text
    print(generated)
