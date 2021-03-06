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

Parameters Used:

Parameter: Number of Hidden Layers
- Value: 2
- Reasoning: Deep neural network is neccessary for complex data and low computation time

Parameter: Architecture (hidden nodes 1, hidden nodes 2)
- Value: (80, 30)
- Reasoning: First layer has two times the number of inputs (43), smaller second layer gives us shorter computation time

Parameter: Hidden Layer Activation Function
- Value: Relu
- Reasoning: Inexpensive training with good performance

Parameter: Number of Output Nodes
- Value: 1
- Reasoning: Model is binary classifier and should have one output predicting if IS_SUCCESSFUL is True or False

Parameter: Output Layer Activation Function
- Value: Sigmoid
- Reasoning: Provides a probability output (value between 0 and 1) for the classification of IS_SUCCESSFUL

From the Base Model before optimization is performed, we get the following results:

Results:
- Loss: 0.549318253993988
- Accuracy: 0.7359766960144043

268/268 - 0s - loss: 0.5493 - accuracy: 0.7360


### Optimization of the Model

Optimization was then performed on the base model by adjusting the parameters above.

Parameter: Training Duration (epochs)
- Change: Increase from 100 epochs to 200 epochs
- Reasoning: Longer training time and could result in more trends learned

Results:
- Loss: 0.5604199767112732
- Accuracy: 0.728863000869751

268/268 - 0s - loss: 0.5604 - accuracy: 0.7289

Parameter: Hidden Layer Activation Function
- Change: Change from Relu to Tanh
- Reasoning: Scaled data results in negative inputs which tanh does not output as zero

Results:
- Loss: 0.5528029799461365
- Accuracy: 0.7285131216049194

268/268 - 0s - loss: 0.5528 - accuracy: 0.7285

Parameter: Number of Input Features
- Change: Reduce from 43 to 34 by bucketing INCOME_AMT and AFFILIATION and dropping the redundant column SPECIAL_CONSIDERATIONS_N after encoding
- Reasoning: Less noise in the input data

Results:
- Loss: 0.5582584738731384
- Accuracy: 0.7261807322502136

268/268 - 0s - loss: 0.5583 - accuracy: 0.7262

________________________________________________________________________________________________________________________________________________________________________

## Summary

In summary, the highest accuracy we achieved was 73% which was achieved with the base deep learning model and before optimization techniques were performed.  This did not meet our target of 75% accuracy and the optimization methods performed on the base deep learning model did not achieve significant improvement.

#### Alternative Model

An alternative to the deep learning model utalized for this project, one could use a Random Forest Classifier.  

A Random Forest Classifier is also appropriate for this binary classification problem and can perform comparably to deep learning models with just two hidden layers.  It is beneficial in that there are fewer parameters to optimize and the parameters that aquire attention are more intuitive than those in a neural network.
