'''
# Jorge Eugenio del Bosque Trevino
# Machine Learning Recipes. K-Fold Cross Validation 
# Evaluate Performance of Machine Learning Algorithms
# Pandas : Read a CSV file, 
# Scikit-Learn : , Logistic Regression, K-Fold (data splitting into K folds)

1. Train and Test Sets 
2. K-fold cross validation <- THIS 
3. Leave One Out Cross-Validation
4. Repeated Random Test-Train Splits

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

### Evaluate using Cross Validation

## import the required libraries
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

## define the variables
filename = 'pima-indians-diabetes.data.csv'
names = ['pregnancies', 'glucose', 'blood pressure', 'skin thickness', 'insulin', 'body mass', 'pedigree', 'age', 'outcome']

# read the csv file with the help of pands and assign it the variable named dataframe
dataframe = read_csv(filename, names = names)

# apply the values method to the dataframe to get only the values and assign them to a new variable named array
array = dataframe.values

# (rows of the array: all), (columns of the array: from the first one to the penultimate or last but one or second last. We don't want the outcome, just the features from which we will prdict)
X = array[:,0:8]

# (all the rows of the array),(just the last column of the array)
Y = array[:,8]

# call the KFold function and tell it to split the dataset in 10 and use 7 as random state
kfold = KFold(n_splits=10, random_state=7)

# creating a variable which contains the definition of which model to use and which solver to apply during execution of the model 
model = LogisticRegression(solver='liblinear')

# calculate the score using the Cross Validation Method from the Sci-Kit Learn Library. Send the parameters of: model, X, Y, and cv=kfold[?]
results = cross_val_score(model, X, Y, cv=kfold)

# two place holders to show the accuracies mean and the standard deviation
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
