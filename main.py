from kmodel import *
import pandas as pd

# Load Data
df = pd.read_csv(r'diabetes.csv')

# Split into train and test
df_train = df[:500].copy()
df_test = df[501:].copy()

# Features columns
features_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Train the model
X_train = df_train[features_columns].values
y_train = df_train['Outcome'].values

# Predictions
X_test = df_test[features_columns].values
y_test = df_test['Outcome'].values


# Inherit from KModel class
model1 = KModel(500, 100, 10, 10, 20, X_train, y_train, X_test, y_test, len(features_columns))
model2 = KModel(500, 100, 10, 10, 40, X_train, y_train, X_test, y_test, len(features_columns))
model3 = KModel(500, 100, 10, 10, 60, X_train, y_train, X_test, y_test, len(features_columns))
model4 = KModel(500, 100, 10, 10, 80, X_train, y_train, X_test, y_test, len(features_columns))

# Fit and evalute the model
model1_accuracy = model1.fit_evaluate_model()
model2_accuracy = model2.fit_evaluate_model()
model3_accuracy = model3.fit_evaluate_model()
model4_accuracy = model4.fit_evaluate_model()
print(f"\nModel1 accuracy: {model1_accuracy}\nModel2 accuracy: {model2_accuracy }\nModel3 accuracy: {model3_accuracy}"
      f"\nModel4 accuracy: {model4_accuracy}")

