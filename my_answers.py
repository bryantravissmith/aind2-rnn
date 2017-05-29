import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # Iterate creating a window table of size windowsize, 
    # but iterate through the data by step_size
    inputs = [text[i:(i+window_size)] for i in range(0, len(text) - window_size, step_size)]
    # get the next text value at the end of each step
    outputs = [text[i] for i in range(window_size, len(text), step_size)]
    
    return inputs,outputs

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # TODO: build an RNN to perform regression on our time series input/output data
    # Create a sequential model 
    model = Sequential()
    # Add LSTM with 200 hidden nodes as stated above
    model.add(LSTM(200, input_shape=(window_size, len(chars))))
    # Predict the character
    model.add(Dense(len(chars)))
    # Provide an activation using softmax
    model.add(Activation('softmax'))

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    from collections import Counter
    cnt = Counter(text)

    # remove as many non-english characters and character sequences as you can 

    import re
    # keep lowercase characters plus space, eclamation mark, comma, period, colon, semicolon, question mark
    p = re.compile("""[a-z !,.:;?]""")
    for l in sorted(cnt.keys()):
        m = p.match(l)
        # if not a regex match of approved characters - replace with space
        if not m:
            text = text.replace(l," ")
    
    # shorten any extra dead space created above
    text = text.replace('  ',' ')

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    # Iterate creating a window table of size windowsize, but iterate through the data by step_size
    inputs = [text[i:(i+window_size)] for i in range(0, len(text) - window_size, step_size)]
    # get the next text value at the end of each step
    outputs = [text[i] for i in range(window_size, len(text), step_size)]
    
    return inputs,outputs
