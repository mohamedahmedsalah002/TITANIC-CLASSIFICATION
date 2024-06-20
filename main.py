import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt

# Load the dataset
file_path = '/Users/mo/Downloads/Titanic-Dataset.csv.xls'  # Replace with the actual path to your file
titanic_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(titanic_data.head())
print(titanic_data.describe())

# Drop irrelevant columns
titanic_data_cleaned = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Handle missing values
imputer = SimpleImputer(strategy='median')
titanic_data_cleaned['Age'] = imputer.fit_transform(titanic_data_cleaned[['Age']])
titanic_data_cleaned['Embarked'].fillna(titanic_data_cleaned['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
titanic_data_cleaned['Sex'] = label_encoder.fit_transform(titanic_data_cleaned['Sex'])
titanic_data_cleaned['Embarked'] = label_encoder.fit_transform(titanic_data_cleaned['Embarked'])

# Split the data into features and target
X = titanic_data_cleaned.drop(columns=['Survived'])
y = titanic_data_cleaned['Survived']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForestClassifier with the specified parameters
rf_model = RandomForestClassifier(n_estimators=1000, random_state=1, criterion='entropy', bootstrap=True, oob_score=True, verbose=1)
rf_model.fit(X_train, y_train)

# Train a LogisticRegression model
lr_model = LogisticRegression(max_iter=1000, random_state=42,)
lr_model.fit(X_train, y_train)

# Make predictions with both models
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)
y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

# Evaluate the RandomForestClassifier on the training set
train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf)
train_precision_rf = precision_score(y_train, y_train_pred_rf)
train_recall_rf = recall_score(y_train, y_train_pred_rf)
train_f1_rf = f1_score(y_train, y_train_pred_rf)

# Evaluate the RandomForestClassifier on the test set
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
test_precision_rf = precision_score(y_test, y_test_pred_rf)
test_recall_rf = recall_score(y_test, y_test_pred_rf)
test_f1_rf = f1_score(y_test, y_test_pred_rf)

# Evaluate the LogisticRegression on the training set
train_accuracy_lr = accuracy_score(y_train, y_train_pred_lr)
train_precision_lr = precision_score(y_train, y_train_pred_lr)
train_recall_lr = recall_score(y_train, y_train_pred_lr)
train_f1_lr = f1_score(y_train, y_train_pred_lr)

# Evaluate the LogisticRegression on the test set
test_accuracy_lr = accuracy_score(y_test, y_test_pred_lr)
test_precision_lr = precision_score(y_test, y_test_pred_lr)
test_recall_lr = recall_score(y_test, y_test_pred_lr)
test_f1_lr = f1_score(y_test, y_test_pred_lr)

# Print evaluation metrics
print("RandomForestClassifier Training Set Evaluation:")
print("Accuracy:", train_accuracy_rf)
print("Precision:", train_precision_rf)
print("Recall:", train_recall_rf)
print("F1 Score:", train_f1_rf)

print("\nRandomForestClassifier Test Set Evaluation:")
print("Accuracy:", test_accuracy_rf)
print("Precision:", test_precision_rf)
print("Recall:", test_recall_rf)
print("F1 Score:", test_f1_rf)

print("\nLogisticRegression Training Set Evaluation:")
print("Accuracy:", train_accuracy_lr)
print("Precision:", train_precision_lr)
print("Recall:", train_recall_lr)
print("F1 Score:", train_f1_lr)

print("\nLogisticRegression Test Set Evaluation:")
print("Accuracy:", test_accuracy_lr)
print("Precision:", test_precision_lr)
print("Recall:", test_recall_lr)
print("F1 Score:", test_f1_lr)

# Check for overfitting in RandomForestClassifier
if train_accuracy_rf > test_accuracy_rf and (train_accuracy_rf - test_accuracy_rf) > 0.1:
    print("\noverfitting")
else:
    print("\nRandomForestClassifier does not appear to be overfitting.")

# Check for overfitting in LogisticRegression
if train_accuracy_lr > test_accuracy_lr and (train_accuracy_lr - test_accuracy_lr) > 0.1:
    print("\noverfitting")
else:
    print("\nLogisticRegression does not appear to be overfitting.")
# Confusion matrix for test set for both models
conf_matrix_rf = confusion_matrix(y_test, y_test_pred_rf)
conf_matrix_lr = confusion_matrix(y_test, y_test_pred_lr)
print("\nRandomForestClassifier Confusion Matrix:\n", conf_matrix_rf)
print("\nLogisticRegression Confusion Matrix:\n", conf_matrix_lr)

# Visualizations with Seaborn
# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = titanic_data_cleaned.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Pair plot
sns.pairplot(titanic_data_cleaned, hue='Survived', palette='bwr', markers=["o", "s"])
plt.suptitle('Pair Plot of Features Colored by Survival', y=1.02)
plt.show()

# Count plot of Survived
sns.countplot(x='Survived', data=titanic_data_cleaned, palette='bwr')
plt.title('Count of Survived')
plt.show()

# Bar plot of survival rate by Sex
sns.barplot(x='Sex', y='Survived', data=titanic_data_cleaned, palette='bwr')
plt.title('Survival Rate by Sex')
plt.show()

# Bar plot of survival rate by Pclass
sns.barplot(x='Pclass', y='Survived', data=titanic_data_cleaned, palette='bwr')
plt.title('Survival Rate by Passenger Class')
plt.show()
