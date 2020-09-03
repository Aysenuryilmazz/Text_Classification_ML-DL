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


# Importing the  inputs
path_models = "Pickles/"

# keras model
path_model = path_models + 'glove_cnn.h5'
cnn = load_model(path_model)

#keras tokenizer
path_tokenizer = path_models + 'keras-tokenizer.pkl'
with open(path_tokenizer, 'rb') as data:
    tokenizer = pickle.load(data)
    
# global variables
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
        
punk_num = string.punctuation + "1234567890"
stop_words_list = list(stopwords.words('english'))

# Definition of functions

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

# Dash App
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

#####
# Edit from here
#####

# Colors
colors = {
    'background': '#ffffff',
    'text': '#696969',
    'header_table': '#ffedb3'
}

# Markdown text
markdown_text1 = '''
This application takes string as input, predicts its diagnosis between 11 categories **ADHD**, **Acne**, **Anxiety**, **Bipolar Disorder**, **Birth Control**, **Depression**, **Insomnia**, **Obesity**, **Pain**, **Weight Loss** and **Other** and then shows a graphic summary.
The news categories are predicted with GloVe + CNN model. 

*Warning: Predictions takes approximately 10 seconds to run necessary calculations.*  
Please enter a text which describes your condition and press **Submit**:
'''

markdown_text2 = '''
*The input text is converted into a pretrained word embedding model. Then, a CNN Classifier is applied to predict its category. For implementation see* [https://www.kaggle.com/aysenur95/text-classification-4-7-gloveandcnn](https://www.kaggle.com/aysenur95/text-classification-4-7-gloveandcnn).

Created by Aysenur Yılmaz.  
kaggle: [@aysenur95](https://www.kaggle.com/aysenur95/notebooks)  
github: [@Aysenuryilmazz](https://github.com/Aysenuryilmazz)  
linkedin: [linkedin.com/in/yilmaz-aysenur](https://www.linkedin.com/in/yilmaz-aysenur/)  

'''

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[

    # Title
    html.H1(children='Condition Prediction',
            style={
                'textAlign': 'left',
                'color': colors['text'],
                'padding': '20px',
                'backgroundColor': colors['header_table']

            },
            className='banner',
            ),

    # Sub-title Left
    html.Div([
        dcc.Markdown(children=markdown_text1)],
        style={'width': '49%', 'display': 'inline-block'}),

    # Sub-title Right
    html.Div([
        dcc.Markdown(children=markdown_text2)],
        style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),

    # Space between text and dropdown
    html.H1(id='space', children=' '),

    # Input text
    html.Div(
        children = dcc.Input( placeholder="i.e. --> I feel like a pain inside my brain. I don't have energy to do anything and i feel lost.", id='input-1-state', type='text',style={'width': '100%'}),
        style={'width': '40%', 'display': 'inline-block', 'float': 'left'}),

    # Button
    html.Div(
        children = [html.Button('Submit', id='submit', type='submit')],
        style={'float': 'center'}
        ),

    # Space between input and output div
    #html.H1(id='space2', children=' '),
    
    # Output Block
    html.Div(id='output-container-button', children='Predicted Contidion is: '),

    # Graph2
    html.Div([
        dcc.Graph(id='graph2')],
        style={'width': '49%', 'float': 'left', 'display': 'inline-block'}),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-value', style={'display': 'none'})

])


@app.callback(
    [Output('intermediate-value', 'children'),
    Output('output-container-button', 'children')],
    [Input('submit', 'n_clicks')],
    [State('input-1-state', 'value')]
   )
def scrape_and_predict(click, values):
    # df_features
    df_features = pd.DataFrame(
         {'text': [values] 
        })


    df_features = df_features.reset_index().drop('index', axis=1)

    # Create predictions array
    predictions_arr = parse_and_predict(df_features)
    # Predict
    prediction = predict_class(predictions_arr)
    # Put into dataset
    clean_df = complete_df(df_features, predictions_arr)
    # pi_df
    pie_df =create_plot_df(predictions_arr)
    return pie_df.to_json(date_format='iso', orient='split'), "Predicted Contidion is: "+prediction


@app.callback(
    Output('graph2', 'figure'),
    [Input('intermediate-value', 'children')])
def update_piechart(jsonified_df):
    df = pd.read_json(jsonified_df, orient='split')


    # Create x and y arrays for the bar plot
    labels = df["name"].tolist()
    values = df["prediction"].tolist()

    # Create plotly figure
    figure = {
        'data': [
            {'values': values,
             'labels': labels,
             'type': 'pie',
             'hole': .4,
             'name': '% tendency of prediction'}
        ],

        'layout': {
            'title': '% tendency of prediction',
            'plot_bgcolor': colors['background'],
            'paper_bgcolor': colors['background'],
            'font': {
                'color': colors['text']
            }
        }

    }

    return figure


# Loading CSS
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})

#####
# To here
#####

if __name__ == '__main__':
    app.run_server(debug=False)