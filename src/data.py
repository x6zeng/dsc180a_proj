import pandas as pd

def standardize_bucket(bucket):
    """
    This function standardized the bucket label and transforms it into a binary output
    """
    if ((bucket == '1.0') | (bucket == '1')):
        return '1'
    elif ((bucket == '2') | (bucket == '3') | (bucket == '2.0') | (bucket == '3.0')):
        return '2 or 3'
    else:
        return bucket

def load_relevant_data(df):
    """
    This function takes in a dataframe, removes duplicates, standardized the bucket labels,
    and returns the text and bucket columns
    """
    df = df[df['country']=='China']
    df = df[['text', 'id', 'Bucket']]
    df['Bucket'] = df['Bucket'].apply(standardize_bucket)
    df_bucket_count = pd.DataFrame(df.groupby('id')['Bucket'].nunique())
    df_bucket_count.reset_index(inplace=True)
    df_bucket_count.columns = ['tweet_id', 'bucket_num']
    bucket_df = df.merge(df_bucket_count, left_on='id', right_on='tweet_id')
    #Remove tweets that are in more than one bucket or has null value for bucket
    bucket_df = bucket_df[bucket_df['bucket_num'] == 1]
    bucket_df = bucket_df[(bucket_df['Bucket'] == '1') | (bucket_df['Bucket'] == '2 or 3')]
    bucket_df = bucket_df.drop_duplicates(subset=['id']).reset_index(drop=True)
    bucket_df = bucket_df[['text', 'Bucket']]
    return bucket_df

def load_sentiment_data(df):
    """
    This function takes in a dataframe, removes duplicates, averages the sentiment score,
    and returns the text and sentiment score columns
    """
    df = df[df['country']=='China']
    df = df[['text', 'id', 'SentimentScore']]
    sent_df = df.copy()[['text', 'id', 'SentimentScore']]
    sent_df.dropna(subset=['SentimentScore'], inplace=True)
    sent_df = pd.DataFrame(sent_df.groupby(['text', 'id'])['SentimentScore'].mean())
    sent_df.reset_index(inplace=True)
    sent_df = sent_df[['text', 'SentimentScore']]
    sent_df = sent_df[sent_df['SentimentScore']<5]
    return sent_df

def load_data(path):
    """
    This function takes in the path of a csv file, loads the csv into a dataframe,
    and returns the relevant and sentiment dataframe
    """
    df = pd.read_csv(path)
    relevant_df = load_relevant_data(df)
    sentiment_df = load_sentiment_data(df)
    return relevant_df, sentiment_df