'''
# Jorge Eugenio del Bosque Trevino
# Machine Learning Recipes. Leave One Out Cross-Validation 
# Evaluate Performance of Machine Learning Algorithms
# Pandas : Read a CSV file, 
# Scikit-Learn : Model Selection : Leave One Out, Cross Validation Score
# Scikit-Learn : Linear Model : Logistic Regression

1. Train and Test Sets 
2. K-fold cross validation
3. Leave One Out Cross-Validation <- THIS 
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

## Evaluate the performance of a Machine Learning Model using Leave One Out Cross-Validation

# import libraries
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# import dataset from csv file
filename = 'pima-indians-diabetes.data.csv'

# create a list that contains the header names of the table
names = ['pregnancies', 'glucose', 'blood pressure', 'skin thickness', 'insulin', 'body mass', 'pedigree', 'age', 'outcome']

# create a data frame that holds just the values of the imported dataset
dataframe = read_csv(filename, names = names)
array = dataframe.values

# (rows of the array: all), (columns of the array: from the first one to the penultimate or last but one or second last. We don't want the outcome, just the features from which we will prdict)
X = array[:,0:8]

# (all the rows of the array),(just the last column of the array)
Y = array[:,8]

# create a variable that holds the name of the function
leaveone = LeaveOneOut()

# create a variable that holds the name of the model we want to run with the desired parameters
model = LogisticRegression(solver ='liblinear')

# calculate results 
results = cross_val_score(model, X, Y, cv=leaveone) 

# show the results
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


