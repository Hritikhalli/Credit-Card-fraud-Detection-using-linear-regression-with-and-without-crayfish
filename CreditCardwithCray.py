import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# Load the credit card dataset
df = pd.read_csv('creditcard.csv')

# Split data into features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

# Combine with Crayfish Optimizer code
class CrayfishOptimizer:
    def __init__(self, objective_function, n_variables, n_crayfish=50, max_iter=1000, alpha=0.5, beta=0.5):
        self.objective_function = objective_function
        self.n_variables = n_variables
        self.n_crayfish = n_crayfish
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta

    def optimize(self):
        crayfish_positions = np.random.rand(self.n_crayfish, self.n_variables)
        best_position = None
        best_fitness = float('inf')

        for _ in range(self.max_iter):
            fitness = self.objective_function(crayfish_positions)
            sorted_indices = np.argsort(fitness)
            if fitness[sorted_indices[0]] < best_fitness:
                best_fitness = fitness[sorted_indices[0]]
                best_position = crayfish_positions[sorted_indices[0]]

            for i in range(self.n_crayfish):
                random_crayfish_index = np.random.choice(sorted_indices[:int(self.beta * self.n_crayfish)])
                crayfish_positions[i] += self.alpha * (crayfish_positions[random_crayfish_index] - crayfish_positions[i])

        return best_position


# Define objective function for Crayfish Optimizer
def objective_function(crayfish_positions):
    fitness = np.zeros(len(crayfish_positions))
    for i, coefficients in enumerate(crayfish_positions):
        model = LinearRegression()
        model.coef_ = coefficients
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        fitness[i] = mean_squared_error(y_val, predictions)
    return fitness

# Create an instance of CrayfishOptimizer
optimizer = CrayfishOptimizer(objective_function, n_variables=X_train.shape[1])

# Optimize to find the best coefficients
best_coefficients = optimizer.optimize()

# Train linear regression model with best coefficients
model = LinearRegression()
model.coef_ = best_coefficients
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

train_precision = precision_score(y_train, np.round(y_train_pred))
val_precision = precision_score(y_val, np.round(y_val_pred))
test_precision = precision_score(y_test, np.round(y_test_pred))

train_recall = recall_score(y_train, np.round(y_train_pred))
val_recall = recall_score(y_val, np.round(y_val_pred))
test_recall = recall_score(y_test, np.round(y_test_pred))

train_f1_score = f1_score(y_train, np.round(y_train_pred))
val_f1_score = f1_score(y_val, np.round(y_val_pred))
test_f1_score = f1_score(y_test, np.round(y_test_pred))

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

