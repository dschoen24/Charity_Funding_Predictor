# Charity_Funding_Predictor
Deep Learning: The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. 

_______________________________________________________________________________________________________________________________________________________

## Overview:

This project uses Jupyter Notebook files that build, train, test and optimize a deep neural network that models charity success from nine features in a loan application data set.  From this dataset, I utalized the features to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

__________________________________________________________________________________________________________________________________________________________

## Results

### Data Preprocessing

The dataset (charity_data.csv) was processed through reading in the data and taking note of the target, the feature and the identification variables:

- Target Variable:  IS_SUCCESSFUL
- Feature Variables: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT
- Identification Variables: EIN, NAME (removed from dataset)

The data was then split into training and testing sets.  The training and testing sets were then scaled using sklearn StandardScaler

### Compiling, Training and Evaluating the Model

Once the data was preprocessed, a base model was built using Tensorflow.keras.models.Sequential & tensorflow.keras.layers.Dense (code can be found in AlphabetSoupCharity.ipynb)



