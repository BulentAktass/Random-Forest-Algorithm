# Importing Libraries
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Reading the csv file
drugs = pd.read_csv("drug200.csv")

# Encode categorical variables
num_Cholesterol = LabelEncoder()
num_BP = LabelEncoder()
num_Sex = LabelEncoder()
num_target = LabelEncoder()

drugs['Cholesterol'] = num_Cholesterol.fit_transform(drugs['Cholesterol'])
drugs['BP'] = num_BP.fit_transform(drugs['BP'])
drugs['Sex'] = num_Sex.fit_transform(drugs['Sex'])
drugs['Drug'] = num_target.fit_transform(drugs['Drug'])

# Define features (X) and target variable (y)
X = drugs.drop(['Drug'], axis=1)
y = drugs["Drug"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=1)

# Fitting and train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

#Number of trees will be displayed:
num_trees_to_visualize = 3

# Visualize the first three trees
for i in range(min(num_trees_to_visualize, len(rf_classifier.estimators_))):
    tree = rf_classifier.estimators_[i]

    plt.figure(figsize=(15, 10))
    plot_tree(tree, feature_names=X.columns, class_names=num_target.classes_, filled=True, rounded=True)
    plt.title(f"Decision Tree {i + 1}")
    plt.show()

#Random Parameter
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,10)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf,param_distributions = param_dist,n_iter=5,cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Generate predictions with the best model
y_pred = best_rf.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

# evaluate the best model with accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Specify 'weighted' for multiclass
recall = recall_score(y_test, y_pred, average='weighted')  # Specify 'weighted' for multiclass

print("New Accuracy:", accuracy)
print("New Precision:", precision)
print("New Recall:", recall)

# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar();
plt.show()

