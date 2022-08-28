import pandas as pd
from tensorflow import keras
from keras.models import Model,Sequential
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from tensorflow import keras
from keras.models import Model,Sequential
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

df=pd.read_csv(r"C:\Users\Asus\OneDrive - IIT Kanpur\Desktop\ml_data\cardio.csv")
#df["price"]=data.target
print(df)
print(df.shape)
print(df.info())
X=df.iloc[:,:10]
y=df.iloc[:,-1]
#X=pd.DataFrame (X,columns= data.feature_names)
#y=pd.DataFrame (y,columns= data.target)
print(X.head())
print(y.head())
from sklearn .model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size= .80)
print(X_train .head())
print(X_test.head())
print(y_train .head())
print(y_test .head())
y_test=pd.DataFrame (y_test )
y_test.reset_index(inplace= True)

from sklearn.linear_model import LogisticRegression
lin=LogisticRegression ()
lin.fit(X_train,y_train)
y_predict=lin.predict(X_test)
y_pred=pd.DataFrame (y_predict)
y_pred.columns=["predicted"]

#y_test .columns=[0,"test"]

print(y_pred.head())
print(y_test.head())
del y_test ["index"]
print(y_test )
from sklearn .metrics import accuracy_score
accuracy_logi=accuracy_score(y_test,y_pred)
print("accuracy=",accuracy_logi )
y_pred =pd.concat ([y_pred,y_test],axis=1)
print(y_pred)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test)
train_data_reshaped=X_train.values.reshape(56000,10,1)
train_label_reshaped=y_train.values.reshape(56000,1,1)
test_data_reshaped=X_test.values.reshape(14000,10,1)
test_label_reshaped=y_test.values.reshape(14000,1,1)
n_timesteps=train_data_reshaped.shape[1]
n_features=train_data_reshaped .shape[2]
print(test_data_reshaped[0].shape)
from keras .layers import LSTM
model=Sequential()
model.add(keras.layers.Input(shape=(n_timesteps ,n_features )))
model.add(keras.layers.Bidirectional(LSTM(10,return_sequences=True, activation="tanh")))
model.add(keras.layers.Bidirectional (LSTM (10,return_sequences=False,activation="tanh")))
model.add(keras.layers.Dense(10,activation="softmax"))
#model.add(Flatten())
model.add(keras.layers.Dense(1,activation="sigmoid"))
model.compile(
    optimizer="Adam",
    loss="binary_crossentropy",
    metrics=['accuracy'])
print(model.summary() )
#history=model.fit(train_data_reshaped ,train_label_reshaped ,epochs= 10)
#print(history)

model.fit(train_data_reshaped,train_label_reshaped ,epochs= 30)
threshold = 0.5

#result = model.predict(padded_docs_test, verbose=2)
#result = result > threshold
y_pred1_nn=model.predict (test_data_reshaped  ).flatten()
y_pred1=y_pred1_nn>threshold
print(y_pred1)
from sklearn .metrics import accuracy_score
accuracy_cnn=accuracy_score(y_test,y_pred1)
print("accuracy=",accuracy_cnn )
"""df["price"]=data.target
print(df)
print(df.shape)
print(df.info())
X=df.iloc[:,:4]
y=df.iloc[:,-1]
#X=pd.DataFrame (X,columns= data.feature_names)
#y=pd.DataFrame (y,columns= data.target)
print(X.head())
print(y.head())
from sklearn .model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size= .80)
print(X_train .head())
print(X_test.head())
print(y_train .head())
print(y_test .head())
y_test=pd.DataFrame (y_test )
y_test.reset_index(inplace= True)
print(df["price"].unique() )

from sklearn.linear_model import LogisticRegression
lin=LogisticRegression ()
lin.fit(X_train,y_train)
y_predict=lin.predict(X_test)
y_pred=pd.DataFrame (y_predict)
y_pred.columns=["predicted"]
#y_test .columns=[0,"test"]

print(y_pred.head())
print(y_test.head())
del y_test ["index"]
print(y_test )
y_pred =pd.concat ([y_pred,y_test],axis=1)
print(y_pred)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test)
train_data_reshaped=X_train.values.reshape(120,4,1)
train_label_reshaped=y_train.values.reshape(120,1,1)
test_data_reshaped=X_test.values.reshape(30,4,1)
test_label_reshaped=y_test.values.reshape(30,1,1)
n_timesteps=train_data_reshaped.shape[1]
n_features=train_data_reshaped .shape[2]
print(test_data_reshaped[0].shape)
model=Sequential()
model.add(keras.layers.Input(shape=(n_timesteps ,n_features )))
model.add(keras.layers.LSTM(150,return_sequences=True, activation="sigmoid"))
model.add(keras.layers.LSTM (150,return_sequences=False,activation="sigmoid"))
model.add(keras.layers.Dense(100,activation="sigmoid"))
#model.add(Flatten())
model.add(keras.layers.Dense(1,activation="sigmoid"))
model.compile(
    optimizer="Adam",
    loss="binary_crossentropy",
    metrics=['accuracy'])
print(model.summary() )
#history=model.fit(train_data_reshaped ,train_label_reshaped ,epochs= 10)
#print(history)

model.fit(train_data_reshaped,train_label_reshaped ,epochs= 10)
threshold = 0.5

#result = model.predict(padded_docs_test, verbose=2)
#result = result > threshold
y_pred1_nn=model.predict (test_data_reshaped  ).flatten()
y_pred1=y_pred1_nn>threshold"""
#print(y_pred1)