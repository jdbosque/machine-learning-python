'''
# Jorge Eugenio del Bosque Trevino
# Machine Learning Snippets   
# Evaluate Performance of Machine Learning Algorithms
# Pandas : Read a CSV file, 
# Scikit-Learn : Logistic Regression, Splitting into Train and Test Sets

1. Train and Test Sets <- THIS 
2. K-fold cross validation
3. Leave One Out Cross-Validation
4. Repeated Random Test-Train Splits

'''

### Evaluate using train and test sets

## import libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

## load data
filename = 'pima-indians-diabetes.data.csv'
names = names = ['pregnancy', 'glucose', 'blood pressure', 'skin thickness', 'insulin', 'body mass index', 'pedigree', 'age', 'outcome']
dataframe = read_csv(filename, names=names)
array = dataframe.values

## separate input from outputs
X = array[:,0:8]
Y = array[:,8]

## create a train set and a split set by splitting in 67% and 33%
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

## define the learning model, appliying and evaluating it.
model = LogisticRegression(solver='liblinear')

# instructing the model to fit the data
model.fit(X_train, Y_train)

# evaluating how good it predicts the results
result = model.score(X_test, Y_test)

## print
print("Acccuracy: %.3f%%" % (result*100.0))
