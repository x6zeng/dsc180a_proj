import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer

def preprocessor():
    stop_words = set(stopwords.words('english'))
    preprocessor = ColumnTransformer(
    transformers=[
        ("tf", CountVectorizer(stop_words=stop_words), 'text_cleaned'),
        ("tfidf", TfidfVectorizer(stop_words=stop_words), 'text_cleaned')]
    )
    return preprocessor

def relevance_model(df, param, test_size, random_num, round_place, testing):
    """This function trains a Bernoulli Naive Bayes to predict the 
    relevance label of the twitter data. It will outputs the accuracy,
    precision, recall, and f1 score of the prediction.
    """
    print('Training Bernoulli Naive Bayes to predict for relevance...')
    bucket_df_1 = df[df.Bucket=='1']
    bucket_df_2_3 = df[df.Bucket=='2 or 3']
    df_len = min(bucket_df_1.shape[0], bucket_df_2_3.shape[0])
    # Balance the dataset
    if testing == True:
        b1_resample = resample(bucket_df_1, replace=True, n_samples=df_len, random_state=random_num)
        b2_3_resample = resample(bucket_df_2_3, replace=True, n_samples=df_len, random_state=random_num)
    else:
        b1_resample = resample(bucket_df_1, replace=False, n_samples=param['train_size'], random_state=random_num)
        b2_3_resample = resample(bucket_df_2_3, replace=False, n_samples=param['train_size'], random_state=random_num)        
    b_df_balanced = pd.concat([b1_resample, b2_3_resample])
    # Train test split
    train, test = train_test_split(b_df_balanced, random_state=random_num, test_size=test_size, shuffle=True)
    X_train = train[['text_cleaned']]
    X_test = test[['text_cleaned']]
    Y_train = train[['Bucket']]
    Y_test = test[['Bucket']]
    # Train the model
    naive_bayes_pipeline = Pipeline([
                    ('preprocessor', preprocessor()),
                    ('clf', BernoulliNB(fit_prior=param['fit_prior']))
                ])
    naive_bayes_pipeline.fit(X_train, Y_train)
    prediction = naive_bayes_pipeline.predict(pd.DataFrame(X_test))
    # Calculate model results
    accuracy = round(accuracy_score(Y_test, prediction), round_place)
    f1 = round(f1_score(np.array(Y_test), prediction, pos_label='1'), round_place)
    precision = round(precision_score(np.array(Y_test), prediction, pos_label='1', average='binary'), round_place)
    recall = round(recall_score(np.array(Y_test), prediction, pos_label='1', average='binary'), round_place)
    print('Accuracy is {}'.format(accuracy))
    print('F1 is {}'.format(f1)) 
    print('Precision is {}'.format(precision)) 
    print('Recall is {}'.format(recall))
    # Output the model results
    output = dict()
    output['Task'] = 'Predict for relevance'
    output['Model'] = 'Bernoulli Naive Bayes'
    if testing==True:
        output['Training Size'] = str(df_len)
    else:
        output['Training Size'] = '4000'
    output['Accuracy'] = accuracy
    output['F1_score'] = f1
    output['Precision'] = precision
    output['Recall'] = recall
    print('Training Bernoulli Naive Bayes to predict for relevance Finished')
    return output

def sentiment_model(df, param, test_size, random_num, round_place, testing):
    """This function trains a Random Forest to predict the 
    sentiment score of the twitter data. It will outputs the MSE and R2
    of the prediction.
    """
    print('Training Random Forest to predict for sentiment...')
    # Train test split
    train, test = train_test_split(df, random_state=random_num, test_size=test_size, shuffle=True)
    X_train = train[['text_cleaned']]
    X_test = test[['text_cleaned']]
    Y_train = train[['SentimentScore']]
    Y_test = test[['SentimentScore']]
    # Train the model
    random_forest_pipeline = Pipeline([
                    ('preprocessor', preprocessor()),
                    ('random_forest', RandomForestRegressor(max_depth=param['max_depth'], 
                                                            min_samples_leaf=param['min_samples_leaf'], 
                                                            min_samples_split=param['min_samples_split'], 
                                                            min_weight_fraction_leaf=param['min_weight_fraction_leaf'], 
                                                            n_estimators=param['n_estimators'], 
                                                            warm_start=param['warm_start'],
                                                            random_state=param['random_state'],
                                                            n_jobs=param['n_jobs']))
            ])
    random_forest_pipeline.fit(X_train, Y_train)
    prediction = random_forest_pipeline.predict(pd.DataFrame(X_test))
    # Calculate model results
    mse = round(mean_squared_error(Y_test, prediction), round_place)
    r2 = round(r2_score(Y_test, prediction), round_place)
    print('MSE is {}'.format(mse))
    print('R2 is {}'.format(r2))
    # Output the model results
    output = dict()
    output['Task'] = 'Predict for sentiment score'
    output['Model'] = 'Random Forest Regressor'
    if testing==True:
        output['Training Size'] = str(df.shape[0])
    else:
        output['Training Size'] = '9022'
    output['MSE'] = mse
    output['R_sqaured'] = r2
    print('Training Random Forest to predict for sentiment Finished')
    return output