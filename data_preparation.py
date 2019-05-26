import numpy as np
import pandas as pd
import matplotlib as mpl

titanic_df = pd.read_csv('data/train.csv',',')

def isMale(row):
    if row['Sex'] == "male":
        return 1
    return 0


def isFemale(row):
    if row['Sex'] == "female":
        return 1
    return 0

def isAlone(row):
    if row['SibSp'] == 0 and row['Parch'] == 0:
        return 1
    return 0

def detectClass(row, Pclass):
    if row['Pclass'] == Pclass:
        return 1
    return 0


# Add male as binary
titanic_df['Male'] = titanic_df.apply(lambda row: isMale(row), axis=1)

# Add female as binary
titanic_df['Female'] = titanic_df.apply(lambda row: isFemale(row), axis=1)

# Create binary information if passanger was alone
titanic_df['Alone'] = titanic_df.apply(lambda row: isAlone(row), axis=1)

# Splits class information into 3 binary columns
titanic_df['First_class'] = titanic_df.apply(lambda row: detectClass(row, 1), axis=1)
titanic_df['Second_class'] = titanic_df.apply(lambda row: detectClass(row, 2), axis=1)
titanic_df['Third_class'] = titanic_df.apply(lambda row: detectClass(row, 3), axis=1)
titanic_df['Third_class'] = titanic_df.apply(lambda row: detectClass(row, 3), axis=1)

# Drop unnecessary columns
titanic_df = titanic_df.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'Sex', 'Pclass','SibSp','Parch'], axis=1)


#Fill NaN with mean
titanic_df =  titanic_df.fillna(titanic_df.mean())

# Drop rows with NaN
#titanic_df = titanic_df[np.isfinite(titanic_df['Age'])]
#titanic_df = titanic_df[np.isfinite(titanic_df['Fare'])]
#titanic_df = titanic_df[np.isfinite(titanic_df['SibSp'])]
#titanic_df = titanic_df[np.isfinite(titanic_df['Parch'])]

# Randomize order 
titanic_df =  titanic_df.sample(frac=1).reset_index(drop=True)

# Normalize Age, SibSp, Parch, Fare
normalized_df=(titanic_df-titanic_df.min())/(titanic_df .max()-titanic_df.min())

titanic_df['Age'] = normalized_df['Age']
#titanic_df['SibSp'] = normalized_df['SibSp']
#titanic_df['Parch'] = normalized_df['Parch']
titanic_df['Fare'] = normalized_df['Fare']

# Splits to test and training set
training_df = titanic_df.iloc[:700,:]
test_df = titanic_df.iloc[700:,:]

# export submit
#titanic_df.to_csv('data/prepared_all.csv')

titanic_df.to_csv('data/prepared_all.csv')
training_df.to_csv('data/prepared_training.csv')
test_df.to_csv('data/prepared_test.csv')


print(training_df.head())
print(training_df.describe())
