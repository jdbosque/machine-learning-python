'''
# Jorge Eugenio del Bosque Trevino
# Machine Learning Recipes. Shuffle Split Cross-Validation 
# Evaluate Performance of Machine Learning Algorithms
# Pandas : Read a CSV file, 
# Scikit-Learn : Model Selection : Shuffle Split, Cross Validation Score
# Scikit-Learn : Linear Model : Logistic Regression

1. Train and Test Sets 
2. K-fold cross validation
3. Leave One Out Cross-Validation
4. Repeated Random Test-Train Splits <- THIS 

'''

'''
# About the Pima Indians Dataset
Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure:  Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Class variable (0 or 1) 268 of 768 are 1, the others are 0
'''
### Evaluate performance of a machine learning alogorithm using Shuffle Split Cross Validation

## import the required libraries
from pandas import read_csv
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

## import the dataset from the csv file
filename = 'pima-indians-diabetes.data.csv'

## create a phython list that contains the labels of the columns or columns headers
names = ['pregnancies', 'glucose', 'blood pressure', 'skin thickness', 'insulin', 'body mass', 'pedigree', 'age', 'outcome']

## create a pandas dataframe with the dataset from the csv file
dataframe = read_csv(filename, names=names)

## create an array that holds the dataframe's values
array = dataframe.values

# (rows of the array: all), (columns of the array: from the first one to the penultimate or last but one or second last. We don't want the outcome, just the features from which we will prdict)
X = array[:,0:8]

# (all the rows of the array),(just the last column of the array)
Y = array[:,8]

## configure the parameters that we will pass to the Shuffle Split function
n_splilts = 10
test_size = 0.33
seed = 7

# configure the variable kfold which holds the ShuffleSplilt parameters
kfold = ShuffleSplit(n_splilts, test_size=test_size,random_state=seed)

## set the model and its parameters and assign them to the variable named model
model = LogisticRegression(solver='liblinear')

## execute the Cross Validation function with the Model, validation set, test set and kfold) 
results = cross_val_score(model, X, Y, cv=kfold)

# show the results
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

