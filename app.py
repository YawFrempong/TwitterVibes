import pickle
import pandas as pd
import preprocessor as cleaner
from pywebio.output import *
from pywebio.input import * 
import snscrape.modules.twitter as sntwitter
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tweets(username, num, table, model, vectorizer):
    vibes = ['☹️','🙂']
    negative_count = 0
    positive_count = 0
    try:
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:' + username).get_items()):
            if i > num:
                break
            
            text = tweet.content
            inference_result = inference(text, model, vectorizer)

            if inference_result == 0:
                negative_count += 1
            elif inference_result == 1:
                positive_count += 1
            
            table.append([text, vibes[inference_result]])
        
        table.append(['Total Positive:' + str(positive_count), 'Total Negative:' + str(negative_count)])
        return True
    except:
        return False

def vectorizer_setup():
    df = pd.read_csv('twitter30k_cleaned.csv')
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    vectorizer.fit_transform(df['twitts'])
    return vectorizer

def inference(text, model, vectorizer):
    clean_text = cleaner.clean(text)
    input_vector = vectorizer.transform([clean_text])
    output = model.predict(input_vector)
    return output[0]

def check_num(num):
    if num < 1:
        return 'You need to fetch at least one tweet'
    if num > 100:
        return 'No more than 100 tweets'


pretrained_model = pickle.load(open('sentiment_model', 'rb'))
vectorizer = vectorizer_setup()
continue_flag = True
while continue_flag:
    with use_scope('homepage', clear=True): 
        put_markdown('## Twitter Vibes')

        input_data = input_group("Basic info",[
            input('Input a twitter handle', name='username', placeholder='@username'),
            input('Input the number of latest twitter to fetch(1-100)', name='num', type=NUMBER, validate=check_num)
        ])

        tweet_data = [['tweet', 'vibe']]
        get_tweets(input_data['username'], input_data['num'], tweet_data, pretrained_model, vectorizer)
        if len(tweet_data) == 1:
            tweet_data.append(['User:' + input_data['username'], 'Not Found'])

        put_table(tweet_data)

        continue_flag = actions(
            label="Would you like to check the vibe of another user?", 
            buttons=[
                {'label': 'Yes', 'value': True}, 
                {'label':'No', 'value': False}
            ]
        )