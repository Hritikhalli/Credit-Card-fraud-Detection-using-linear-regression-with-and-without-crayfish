import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Load the credit card dataset
data = pd.read_csv('creditcard.csv')
df = pd.DataFrame(data)      
number_fraud = len(data[data.Class == 1])
number_no_fraud = len(data[data.Class == 0])
print('There are only {} frauds in the original dataset, even though there are {} no frauds in the dataset.'.format(str(number_fraud),str(number_no_fraud)))
df_train_all = df[0:150000] # dividing the dataset into two parts
df_train_1 = df_train_all[df_train_all['Class'] == 1] # Fraud
df_train_0 = df_train_all[df_train_all['Class'] == 0] # Non Fraud
print('In this dataset, we have {} frauds so we need to take a similar number of non-fraud'.format(len(df_train_1)))

df_sample=df_train_0.sample(300) # No Fraud
df_train = df_train_1.append(df_sample) # We gather the frauds with the no frauds.
df_train = df_train.sample(frac=1) # Then we mix our dataset
X_train = df_train.drop(['Time', 'Class'],axis=1) # We drop the features which are useless like Time ,the labels
y_train = df_train['Class'] # Creating a target class
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
#with all the test dataset to see if the model learn correctly
df_test_all = df[150000:]

X_test = df_test_all.drop(['Time', 'Class'],axis=1)
y_test = df_test_all['Class']
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

# Split data into training, validation, and test sets
X_val, X_test, y_val, y_test = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model performance
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

print("Training R-squared: ", round(r2_score(y_train, y_train_pred), 2))
print("Validation R-squared: ", round(r2_score(y_val, y_val_pred), 2))
print("Test R-squared: ", round(r2_score(y_test, y_test_pred), 2))

train_accuracy = accuracy_score(y_train, np.round(y_train_pred))
val_accuracy = accuracy_score(y_val, np.round(y_val_pred))
test_accuracy = accuracy_score(y_test, np.round(y_test_pred))

train_precision = precision_score(y_train, np.round(y_train_pred), average='micro')
val_precision = precision_score(y_val, np.round(y_val_pred), average='micro')
test_precision = precision_score(y_test, np.round(y_test_pred), average='micro')


train_recall = recall_score(y_train, np.round(y_train_pred), average='micro')
val_recall = recall_score(y_val, np.round(y_val_pred), average='micro')
test_recall = recall_score(y_test, np.round(y_test_pred), average='micro')

train_f1_score = f1_score(y_train, np.round(y_train_pred), average='micro')
val_f1_score = f1_score(y_val, np.round(y_val_pred), average='micro')
test_f1_score = f1_score(y_test, np.round(y_test_pred), average='micro')

train_mcc = matthews_corrcoef(y_train, np.round(y_train_pred))
val_mcc = matthews_corrcoef(y_val, np.round(y_val_pred))
test_mcc = matthews_corrcoef(y_test, np.round(y_test_pred))

# Print metrics
print("Training Metrics:")
print("Accuracy:", train_accuracy)
print("Precision:", train_precision)
print("Recall:", train_recall)
print("F1-Score:", train_f1_score)
print("MCC:", train_mcc)

print("\nValidation Metrics:")
print("Accuracy:", val_accuracy)
print("Precision:", val_precision)
print("Recall:", val_recall)
print("F1-Score:", val_f1_score)
print("MCC:", val_mcc)

print("\nTest Metrics:")
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1-Score:", test_f1_score)
print("MCC:", test_mcc)

# Plot actual vs predicted values
df_pred = pd.DataFrame(y_val.values, columns=['Actual'], index=y_val.index)
df_pred['Predicted'] = y_val_pred
df_pred.plot()
plt.show()
