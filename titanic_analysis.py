# ============================================================
# Titanic Survival Analysis & Logistic Regression
# ============================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

train = pd.read_csv('titanic_dataset.csv')

print("Dataset shape:", train.shape)
print("\nFirst 5 rows:")
print(train.head())

print("\nDataset information:")
train.info()


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

sns.set_style('whitegrid')


# Missing values
plt.figure(figsize=(10, 6))

sns.heatmap(
    train.isnull(),
    yticklabels=False,
    cbar=False,
    cmap='viridis'
)

plt.title('Missing Values in Titanic Dataset')
plt.tight_layout()
plt.show()


# Survival distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    x='Survived',
    hue='Survived',
    data=train,
    legend=False
)

plt.title('Survival Distribution')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Number of Passengers')
plt.tight_layout()
plt.show()


# Survival by sex
plt.figure(figsize=(7, 5))

sns.countplot(
    x='Survived',
    hue='Sex',
    data=train,
    palette='RdBu_r'
)

plt.title('Survival by Sex')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Number of Passengers')
plt.tight_layout()
plt.show()


# Survival by passenger class
plt.figure(figsize=(7, 5))

sns.countplot(
    x='Survived',
    hue='Pclass',
    data=train,
    palette='rainbow'
)

plt.title('Survival by Passenger Class')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Number of Passengers')
plt.tight_layout()
plt.show()


# Age distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    train['Age'].dropna(),
    kde=False,
    color='darkred',
    bins=40
)

plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.tight_layout()
plt.show()


# ============================================================
# 4. DATA CLEANING
# ============================================================

# Analyze passenger age by class before filling missing Age values
plt.figure(figsize=(10, 6))

sns.boxplot(
    x='Pclass',
    y='Age',
    hue='Pclass',
    data=train,
    palette='winter',
    legend=False
)

plt.title('Age Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.tight_layout()
plt.show()


# Fill missing Age values based on passenger class
def impute_age(cols):
    age = cols['Age']
    pclass = cols['Pclass']

    if pd.isnull(age):

        if pclass == 1:
            return 37

        elif pclass == 2:
            return 29

        else:
            return 24

    return age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

# Fill missing Embarked values with the most common port
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

# Cabin contains too many missing values, so it is removed
train.drop('Cabin', axis=1, inplace=True)

# Check remaining missing values
print("\nMissing values after cleaning:")
print(train.isnull().sum())


# ============================================================
# 5. CATEGORICAL DATA ENCODING
# ============================================================

# Convert categorical variables into dummy variables
sex = pd.get_dummies(
    train['Sex'],
    drop_first=True,
    dtype=int
)

embark = pd.get_dummies(
    train['Embarked'],
    drop_first=True,
    dtype=int
)


# Remove original categorical and unnecessary columns
train.drop(
    ['PassengerId', 'Sex', 'Embarked', 'Name', 'Ticket'],
    axis=1,
    inplace=True
)


# Add encoded variables to the dataset
train = pd.concat(
    [train, sex, embark],
    axis=1
)


print("\nProcessed dataset:")
print(train.head())


# ============================================================
# 6. PREPARE FEATURES AND TARGET
# ============================================================

# X contains the features used for prediction
X = train.drop('Survived', axis=1)

# y contains the target variable
y = train['Survived']


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

# Use 70% of the data for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=101
)


# ============================================================
# 8. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

logmodel = LogisticRegression(max_iter=1000)

logmodel.fit(X_train, y_train)


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

predictions = logmodel.predict(X_test)


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

conf_matrix = confusion_matrix(
    y_test,
    predictions
)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n================ MODEL RESULTS ================")

print("\nConfusion Matrix:")
print(conf_matrix)

print(f"\nAccuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# ============================================================
# 11. CONFUSION MATRIX VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False
)

plt.title('Logistic Regression - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()