import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import r2_score
st.title('Used Car for Sale')
st.text('Find the right car as per your choice')
df=pd.read_csv('Car_prediction_app.csv')
st.image('car_image.jpeg')
price_range=st.slider("Price Range",min_value=int(df['price'].min()),max_value=int(df['price'].max()),step=(5000),value=int(df['price'].min()))
number_of_cylinders=st.slider("no_of_cylinders",min_value=int(df['no_of_cylinders'].min()),max_value=int(df['no_of_cylinders'].max()),step=(1),value=int(df['no_of_cylinders'].min()))
mileage_range=st.slider("mileage",min_value=int(df['mileage'].min()),max_value=int(df['mileage'].max()),step=(2),value=int(df['mileage'].min()))
df.loc[(df['price']<=price_range) & (df['no_of_cylinders']<=number_of_cylinders) & (df['mileage']<=mileage_range)]
st.text("Price selected is "+ str(price_range))
st.header('Price predictor')
sel_box_var=st.selectbox("Select Method",['Linear','Ridge','Lasso'],index=0)
multi_var=st.multiselect("Select Additional Variables for accuracy= ",['hp','Wheel_Base','Engine_Size','Bore_Ratio'])
df_new=[]
df_new=df[multi_var]
if sel_box_var=='Linear':
    df_new['mileage']=df['mileage']
    df_new['no_of_cylinders']=df['no_of_cylinders']
    X=df_new
    Y=df['price']
    model=LinearRegression()
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    reg=model.fit(X_train,Y_train)
    Y_pred=reg.predict(X_test)
    st.text("Intercept= " + str(reg.intercept_))
    st.text("Coefficient= " + str(reg.coef_))
    st.text("R^2= " +str(r2_score(Y_test,Y_pred)))
elif sel_box_var=='Lasso':
    df_new['mileage']=df['mileage']
    df_new['no_of_cylinders']=df['no_of_cylinders']
    X=df_new
    Y=df['price']
    model=Lasso()
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    reg=model.fit(X_train,Y_train)
    Y_pred=reg.predict(X_test)
    st.text("Intercept= " +str(reg.intercept_))
    st.text("Coefficient= " +str(reg.coef_))
    st.text("R^2= " +str(r2_score(Y_test,Y_pred)))
else:
    df_new['mileage']=df['mileage']
    df_new['no_of_cylinders']=df['no_of_cylinders']
    X=df_new
    Y=df['price']
    model=Ridge()
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
    reg=model.fit(X_train,Y_train)
    Y_pred=reg.predict(X_test)
    st.text("Intercept= " +str(reg.intercept_))
    st.text("Coefficient= " +str(reg.coef_))
    st.text("R^2= " +str(r2_score(Y_test,Y_pred)))
st.set_option('deprecation.showPyplotGlobalUse',False)
sns.regplot(Y_test,Y_pred)
st.pyplot()
count=0
pred_val=0
for i in df_new.keys():
    try:
        val=st.text_input("Enter no./Val of",+i)
        pred_val=pred_val+int(val)*reg.coef_[count]
        count=count+1
    except:
        pass
st.text('Predicted prices are: ' +str(pred_val+reg.intercept_))
st.header("Upload your queries")
img=st.file_uploader("Upload")
st.text("Details for the representatives to contact you")
st.text("Enter your address")
address=st.text_area("Your address here")
date=st.date_input('Enter a date')
if st.checkbox("I confirm the date and time",value=False):
    st.write("Thanks for confirming")
st.number_input("Rate our site,min_value=1,max_value=10")
