import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import string
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
nltk.download('stopwords')

# global variables
punk_num = string.punctuation + "1234567890"
stop_words_list = list(stopwords.words('english'))

category_codes_rev={    
     0: 'ADHD',
     1: 'Acne',
     2: 'Anxiety',
     3: 'Bipolar Disorde',
     4: 'Birth Control',
     5: 'Depression',
     6: 'Insomnia',
     7: 'Obesity',
     8: 'Pain',
     9: 'Weight Loss'
} 

# definiton of functions

def parse_and_predict(df):    
    df['text_Parsed_1'] = df['text'].str.replace("\r", " ")
    df['text_Parsed_1'] = df['text_Parsed_1'].str.replace("\n", " ")
    df['text_Parsed_1'] = df['text_Parsed_1'].str.replace("    ", " ")
    df['text_Parsed_1'] = df['text_Parsed_1'].str.replace('"', '')

    df['text_Parsed_2'] = df['text_Parsed_1'].str.lower()

    df['text_Parsed_3'] = df['text_Parsed_2']

    for punct_sign in punk_num:
        df['text_Parsed_3'] = df['text_Parsed_3'].str.replace(punct_sign, '')

    df['text_Parsed_4'] = df['text_Parsed_3'].str.replace("'s", "")

    wordnet_lemmatizer = WordNetLemmatizer()
    nrows = len(df)
    lemmatized_text_list = []
    for row in range(0, nrows):

        # Create an empty list containing lemmatized words
        lemmatized_list = []
        # Save the text and its words into an object
        text = df.loc[row]['text_Parsed_4']
        text_words = text.split(" ")
        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        # Join the list
        lemmatized_text = " ".join(lemmatized_list)
        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)

    df['text_Parsed_5'] = lemmatized_text_list

    df['text_Parsed_6'] = df['text_Parsed_5']

    for stop_word in stop_words_list:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['text_Parsed_6'] = df['text_Parsed_6'].str.replace(regex_stopword, '')

    df = df['text_Parsed_6']
    df = pd.DataFrame(df.values, columns=["text_Parsed"] )

    # tokenizer and predict
    seq = tokenizer.texts_to_sequences(df["text_Parsed"])
    pad = pad_sequences(seq, padding='post', maxlen=100)
    pred_arr = cnn.predict(pad)
    
    return pred_arr


def predict_class(pred_arr):
    binary_form = np.where(pred_arr > 0.5, 1, 0)
    if np.sum(binary_form) == 0:
        return "Other"
    else:
        return category_codes_rev[np.argmax(pred_arr)]


def complete_df(df, pred_arr):
    df['Prediction'] = predict_class(pred_arr)
    df2 = df.rename(columns={'text_Parsed_6': 'text_Parsed'})
    df2 = df2.drop(["text_Parsed_1","text_Parsed_2","text_Parsed_3","text_Parsed_4","text_Parsed_5","text"],axis=1)
    return df2


def create_plot_df(pred_arr):
    df = pd.DataFrame(zip(list(category_codes_rev.values()), list(np.squeeze(pred_arr))), columns=["name", "prediction"])
    return df


################################ MAIN ###################################
# change this text to test predictor
input_text = "I feel like a pain inside my brain. I don't have energy to do anything and i feel lost."

# keras model
cnn = load_model('glove_cnn.h5')

#keras tokenizer
with open('keras-tokenizer.pkl', 'rb') as data:
    tokenizer = pickle.load(data)
    
    
# df_text
df_text = pd.DataFrame({'text': [input_text]}, index=[0])

# Create predictions array
predictions_arr = parse_and_predict(df_text)
# Predict
prediction = predict_class(predictions_arr)

print(predictions_arr)
print(prediction)