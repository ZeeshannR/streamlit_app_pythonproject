# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:11:43 2021

@author: DELL
"""

import streamlit as st
import pandas as pd
#import numpy as np
import seaborn as sns
#import plotly.express as px
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error,r2_score
#from sklearn.metrics import r2_score
import plotly.graph_objects as go
import random

st.title('Used Car for Sale')
st.text('Find the right car as per your choice')
df=pd.read_csv('cleaned data.csv')
st.image('car_image.jpeg')
price_range=st.slider("Price Range",min_value=int(df['selling_price'].min()),max_value=int(df['selling_price'].max()),step=(5000),value=int(df['selling_price'].min()))
df['brand'] = df['name'].str.split(' ').str.get(0)
dm = df.loc[(df['selling_price']<=price_range)]
fig=go.Figure(data=[go.Table(header=dict(values=['brand']),cells=dict(values=[dm.brand.unique()]))])
st.write(fig)
st.text("Price selected is "+ str(price_range))
st.header('Price predictor')
sel_box_var=st.selectbox("Select Method",['Linear','Ridge','Lasso'],index=0)
multi_var=st.multiselect("Select Additional Variables for accuracy = ",['seats','mileage','km_driven'])
df_new=[]
df_new=df[multi_var]
if sel_box_var=='Linear':
    df_new['age']=df['age']
    X=df_new
    Y=df['selling_price']
    model=LinearRegression()
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    reg=model.fit(X_train,Y_train)
    Y_pred=reg.predict(X_test)
elif sel_box_var=='Lasso':
    df_new['age']=df['age']
    X=df_new
    Y=df['selling_price']
    model=Lasso()
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    reg=model.fit(X_train,Y_train)
    Y_pred=reg.predict(X_test)
else:
    df_new['age']=df['age']
    X=df_new
    Y=df['selling_price']
    model=Ridge()
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    reg=model.fit(X_train,Y_train)
    Y_pred=reg.predict(X_test)
st.set_option('deprecation.showPyplotGlobalUse',False)
sns.regplot(Y_test,Y_pred)
count=0
pred_val=0
for i in df_new.keys():
    try:
        val=st.text_input("Enter no./Val of "+i)
        pred_val=pred_val+float(val)*reg.coef_[count]
        count=count+1
    except:
       pass
st.text('Predicted prices are: ' +str(pred_val+reg.intercept_))
st.header("Upload your queries")
img=st.file_uploader("Upload")
st.text('What should we call you?')
name=st.text_input("Your name here:")
st.text("Details for the representatives to contact you")
st.text("Enter your address")
address=st.text_area("Your address here")
date=st.date_input('Enter a date')

if st.checkbox("I confirm the date and time",value=False):
    if st.button('Register!'):
        userdict = {"Name": name, "Address":address, "Date":date}
        userdf = pd.DataFrame(userdict.items())
        userdf.to_excel(name + str(random.randint(1, 1000)) + ".xlsx")
        st.write("Thanks for registering with us! We will get back to you on the mentioned details. Thanks for connecting.")
st.number_input("Rate our site,min_value=1,max_value=10")
