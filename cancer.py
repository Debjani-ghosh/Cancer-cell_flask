import pandas as pd
import numpy as np
import pickle
df=pd.read_csv('data.csv')
df=df.dropna(axis=1)
from sklearn.preprocessing import LabelEncoder
l_e=LabelEncoder()
df.iloc[:,1]=l_e.fit_transform(df.iloc[:,1].values)
x=np.array(df.iloc[:,2:32])
y=np.array(df.iloc[:,1])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rmc=RandomForestClassifier(max_depth=3,random_state=0)
rmc.fit(x_train,y_train)
pred1=rmc.predict(x_test)
print(pred1)
arr=np.array([[13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]])
r=rmc.predict(arr)
print(r)
sc1=accuracy_score(y_test,pred1)
print(sc1)
pickle.dump(rmc, open('can.pkl', 'wb'))