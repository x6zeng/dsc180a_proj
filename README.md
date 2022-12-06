## DSC180a Quarter 1 Project: Sentiment Analysis on U.S. Congress Twitter Data

## Folders
* config: contains parameters of the models
  - config.json: model parameters and configurations
* data: contains both test data and raw data
  - test: folder to store the test data
    - data.csv: the test data used to test the correctness of the code
  - raw: folder to store the raw data

* notebook: contains Jupyter Notebooks for EDA and model experiments
  - Exploratory_Data_Analysis.ipynb
  - Relevance_Bucket_Model.ipynb
  - Sentiment_Score_Model.ipynb
* src: contains the source code used to run the code
  - data.py: load data from file path
  - model.py: feature engineering, train, test models
  - preprocessing.py: preprocess dataframe

## Raw Data:
Our raw dataset is located in this [Google Drive](https://drive.google.com/drive/folders/1VSYdGh12UNVNhfxbSeHRdANvHr5xF8Ea?usp=share_link). If you cannot access the drive, please contact the team's mentor Dr. Molly Roberts.

To run the project on the raw data, please follow the link above and download SentimentLabeled_10112022.csv, renamed the file as data.csv and saved it under the raw folder under the data folder.

## Run the project
* To run the code on test data:
  - run python run.py test
* To run the code on full data:
  - download the full data following the instruction, rename the file, and save it under the raw folder under the data folder
  - run python run.py
