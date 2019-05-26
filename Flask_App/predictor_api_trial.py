"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.

This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.preprocessing.sequence import pad_sequences

# load the model
model = tf.keras.models.load_model('/Users/tcbon/Desktop/Coding/Metis/Bootcamp/Project_4/Flask_App/static/models/new_test_model_1.hdf5')
 #Flask workaround with Tensorflow error
model._make_predict_function()
graph = tf.get_default_graph()
# load the tokenizer
tokenizer = pickle.load(open('/Users/tcbon/Desktop/Coding/Metis/Bootcamp/Project_4/Flask_App/static/models/new_test_tokenizer_1.pkl', 'rb'))


def some_route(inputs):
    global graph
    with graph.as_default():
        outputs = model.predict_classes(inputs)
    return outputs


# generate a sequence from a language model
def generate_seq(seed_text):
    seq_length = 50
    n_words = 50
    
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        
       

        # predict probabilities for each word
        yhat = some_route(encoded)
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


# This section checks that the generative code runs properly
# To run, type "python predictor_api.py" in the terminal.
#
# The if __name__='__main__' section ensures this code only runs
# when running this file; it doesn't run when importing
if __name__ == '__main__':
    
    seed_text = input('Type a phrase for the machine to complete:')
    
    # generate new text
    generated = generate_seq(seed_text)
    print(generated)
