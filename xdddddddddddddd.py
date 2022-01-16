# %%
import pandas as pd
import numpy as py
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, log_loss, confusion_matrix, f1_score, accuracy_score
import time as time

start_time = time.time()
dataset = pd.read_csv("framingham.csv")
dataset.head()

# We will impute missing data using the mean of the column
preprocess = SimpleImputer(missing_values = py.NaN, strategy = 'mean')
dataset.totChol = preprocess.fit_transform(dataset['totChol'].values.reshape(-1, 1))[:,0]
dataset.BMI = preprocess.fit_transform(dataset['BMI'].values.reshape(-1, 1))[:,0]
dataset.heartRate = preprocess.fit_transform(dataset['heartRate'].values.reshape(-1, 1))[:,0]
dataset.glucose = preprocess.fit_transform(dataset['glucose'].values.reshape(-1, 1))[:,0]
dataset.age = preprocess.fit_transform(dataset['age'].values.reshape(-1, 1))[:,0]

input_features = ['age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
x = dataset[input_features]
y = dataset.TenYearCHD

# Train test split will split 80% for training and 20% of the data for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# We will experiment with three different combinations of parameters and compare them in a table in the report
model_one = MLPClassifier(hidden_layer_sizes = (10, 10, 10, 10, 10, 10, 10), solver = 'adam', max_iter = 100000)
model_two = MLPClassifier(hidden_layer_sizes = (20, 20, 20, 20, 20, 20, 20), solver = 'adam', max_iter = 100000)
model_three = MLPClassifier(hidden_layer_sizes = (300, 300, 300, 300, 300, 300, 300), solver = 'adam', max_iter = 100000)

# We then train the data on each model 
model_one.fit(x_train, y_train)
model_two.fit(x_train, y_train)
model_three.fit(x_train, y_train)

# Model one: hidden_layer_sizes = 100, solver = 'adam', max_iter = 100000
ymodel_one_train = model_one.predict(x_train)
ymodel_one_test = model_one.predict(x_test)
ymodel_one_log_train = model_one.predict_proba(x_train)
ymodel_one_log_test = model_one.predict_proba(x_test)

# Model two: hidden_layer_sizes = 150, solver = 'adam', max_iter = 100000
ymodel_two_train = model_two.predict(x_train)
ymodel_two_test = model_two.predict(x_test)
ymodel_two_log_train = model_two.predict_proba(x_train)
ymodel_two_log_test = model_two.predict_proba(x_test)

# Model three: hidden_layer_sizes = 200, solver = 'adam', max_iter = 100000
ymodel_three_train = model_three.predict(x_train)
ymodel_three_test = model_three.predict(x_test)
ymodel_three_log_train = model_three.predict_proba(x_train)
ymodel_three_log_test = model_three.predict_proba(x_test)

# Model one training data
print("Model one training data:")
print('Train Data:\n', classification_report(y_train, ymodel_one_train))
print('Log Loss:', log_loss(y_train, ymodel_one_log_train))
print('F1 Score:', 1 - f1_score(y_train, ymodel_one_train))
print('Accuracy Score:', accuracy_score(y_train, ymodel_one_train))
print('Confusion Matrix:\n', confusion_matrix(y_train, ymodel_one_train))

# Model one test data
print("\nModel one test data:")
print('Test Data:\n', classification_report(y_test, ymodel_one_test))
print('Log Loss:', log_loss(y_test, ymodel_one_log_test))
print('F1 Score:', 1 - f1_score(y_test, ymodel_one_test))
print('Accuracy Score:', accuracy_score(y_test, ymodel_one_test))
print('Confusion Matrix:\n', confusion_matrix(y_test, ymodel_one_test))

# Model two training data
print("\nModel two training data:")
print('Train Data:\n', classification_report(y_train, ymodel_two_train))
print('Log Loss:', log_loss(y_train, ymodel_two_log_train))
print('F1 Score:', 1 - f1_score(y_train, ymodel_two_train))
print('Accuracy Score:', accuracy_score(y_train, ymodel_two_train))
print('Confusion Matrix:\n', confusion_matrix(y_train, ymodel_two_train))

# Model two test data
print("\nModel two test data:")
print('Test Data:\n', classification_report(y_test, ymodel_two_test))
print('Log Loss:', log_loss(y_test, ymodel_two_log_test))
print('F1 Score:', 1 - f1_score(y_test, ymodel_two_test))
print('Accuracy Score:', accuracy_score(y_test, ymodel_two_test))
print('Confusion Matrix:\n', confusion_matrix(y_test, ymodel_two_test))

# Model three training data
print("\nModel three training data:")
print('Train Data:\n', classification_report(y_train, ymodel_three_train))
print('Log Loss:', log_loss(y_train, ymodel_three_log_train))
print('F1 Score:', 1 - f1_score(y_train, ymodel_three_train))
print('Accuracy Score:', accuracy_score(y_train, ymodel_three_train))
print('Confusion Matrix:\n', confusion_matrix(y_train, ymodel_three_train))

# Model three test data
print("\nModel three test data:")
print('Test Data:\n', classification_report(y_test, ymodel_three_test))
print('Log Loss:', log_loss(y_test, ymodel_three_log_test))
print('F1 Score:', 1 - f1_score(y_test, ymodel_three_test))
print('Accuracy Score:', accuracy_score(y_test, ymodel_three_test))
print('Confusion Matrix:\n', confusion_matrix(y_test, ymodel_three_test))

# Extra information
print("\nIteration count:")
print("Model one: ", model_one.n_iter_)
print("Model two: ", model_two.n_iter_)
print("Model three: ", model_three.n_iter_, '\n')

print("Layer count including input and output layers:")
print("Model one: ", model_one.n_layers_)
print("Model two: ", model_two.n_layers_)
print("Model three: ", model_three.n_layers_)

print('\nRun Time: ', round(time.time() - start_time, 10))

# %%
