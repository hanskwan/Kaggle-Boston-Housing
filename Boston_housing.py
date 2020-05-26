# Boston Housing Prediction
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

# Linear regression module
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler


df = pd.read_csv("housing.csv")
df.info()
df.head(10)
df.isnull().sum()

plt.plot(df["RM"],df["LSTAT"],df["PTRATIO"],"ro")

sns.pairplot(df)

# Features
# RM: average number of rooms per dwelling
# LSTAT: percentage of population considered lower status
# PTRATIO: pupil-teacher ratio by town
# Target Variable 4. MEDV: median value of owner-occupied homes

### Seperate train and test set
X = df.iloc[:, :-1].values
y = pd.DataFrame(df.iloc[:, -1]).values

#Scaling
sc = StandardScaler()
X_sc = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size = 0.20, random_state = 5)

#### 
# linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_train_score =lr.score(X_train, y_train)
lr_test_score =lr.score(X_test, y_test)

reg_score = {"Train Score" : lr_train_score, "Test Score": lr_test_score}
reg_score


####
# MLP model

# Import
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import adam
from keras import metrics
from sklearn.metrics import r2_score

# Buuild MLP
def build_ANN():
    model = Sequential([
            Dense(8, input_dim = 3, activation = 'relu'),
            Dense(units = 16, activation = 'relu'),
            Dense(units = 8, activation = 'relu'),
            Dense(units = 1, activation = 'relu')
            ])
    model.compile(loss = "mse",optimizer = "adam")
    return model
 
ANN1 = build_ANN()
ANN1.summary()

ANN1_ted = ANN1.fit(X_train, y_train, batch_size = 16, epochs = 200)

# MLP evaluation
X_train_pred = ANN1_ted.predict(X_train)
eval_train = r2_score(X_train_pred, y_train)
eval_train

y_pred = ANN1.predict(X_test)
eval_test = r2_score(y_pred,y_test)
eval_test

result = pd.DataFrame({"Training accuracy" : eval_train, "Test accuracy" : eval_test}, index = [0])
result

# Initialising the ANN
MLP_1 = Sequential()
MLP_1.add(Dense(8, input_dim = 3, activation = 'relu'))
MLP_1.add(Dense(units = 16, activation = 'relu'))
MLP_1.add(Dense(units = 8, activation = 'relu'))
MLP_1.add(Dense(units = 1,activation = 'relu')) # Output layer

MLP_1.compile(optimizer = "adam", loss = 'mse')
MLP_1.summary()
# Fitting the ANN to the Training set
MLP_1.fit(X_train, y_train, batch_size = 16, epochs = 200)

# MLP evaluation
X_train_pred = MLP_1.predict(X_train)
eval_train = r2_score(X_train_pred, y_train)
eval_train

y_pred = MLP_1.predict(X_test)
eval_test = r2_score(y_pred,y_test)
eval_test

result = pd.DataFrame({"Training accuracy" : eval_train, "Test accuracy" : eval_test}, index = [0])
result
######

model = Sequential()
model.add(Dense(3, input_dim = 2))
model.add(Dense(1, ))
model.summary()
