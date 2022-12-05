import os
import json
import sys
import warnings
from src.preprocessing import construct_features
from src.model import relevance_model, sentiment_model
from src.data import load_data

def main(targets):
    '''
    The function takes in a target and output the best model results for 
    predicting relevance and sentiment of the twitter data
    '''
    try:
        #Load in data
        print('Load Data begins')
        if 'test' in targets:
            fp = os.path.join('data/test', 'data.csv')
        relevance_df, sentiment_df = load_data(fp)
        print('Load Data finishes')

        #Load Configuration
        print('Load Configuration begins')
        with open('config/config.json') as fh:
            param = json.load(fh)

        relevance_param = param['bernoulli_naive_bayes_param']
        sentiment_param = param['random_forest_param']
        test_size = param['test_size']
        random_num = param['random_num']
        round_place = param['round_place']
        print('Load Configuration finishes')

        #Data preprocessing
        print('Data preprocessing begins')
        relevance_df = construct_features(relevance_df)
        sentiment_df = construct_features(sentiment_df)
        print('Data preprocessing finishes')

        #Train and test models
        print('Train model begins')
        relevance_output = relevance_model(relevance_df, relevance_param, test_size, random_num, round_place)
        sentiment_output = sentiment_model(sentiment_df, sentiment_param, test_size, random_num, round_place)
        print('Train model finishes')
        return [relevance_output, sentiment_output]

    except Exception as e:
        print('Encountered error when running scripts')
        print(e)


if __name__ == '__main__':
    warnings.filterwarnings('ignore') # ignore warnings
    targets = sys.argv[1:]
    model_output = main(targets)
    print(model_output)