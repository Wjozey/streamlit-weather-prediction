import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


header = st.container()
dataset = st.container()
modelTraining = st.container()


@st.cache
def get_data(filename):
    weather_data = pd.read_csv(filename)
    weather_data['TIME'] = weather_data['TIME'].astype(str)
    weather_data['TIME'] = [x[:8] for x in weather_data['TIME']]
    weather_data['TIME'] = pd.to_datetime(weather_data['TIME'])
    weather_data.rename(columns= {'TIME' :'datetime', 'TEMPERATURE [degC]' : 'Temp', 'PRECIPITATION [mm/6hr]' : 'precipitation', 'WIND SPEED [m/s]' : 'wind_speed', 'WIND DIRECTION' : 'wind_direction', 'HUMIDITY [%]': 'humidity', 'SEA-LEVEL PRESSURE [hPa]' : 'pressure'}, inplace=True)
    return weather_data

with header:
    st.title('Interactive Temperature Prediction App !')
    st.text('In this project I use a Random Regressor Model to predict temperature')

with dataset:
    st.header('Weather dataset - Ulsan, Korea from 1980 to 2018')
    st.write('dataset source [here]("https://raw.githubusercontent.com/PotatoThanh/Bidirectional-LSTM-and-Convolutional-Neural-Network-For-Temperature-Prediction/master/data/data.csv)')

    weather_data = get_data('https://raw.githubusercontent.com/PotatoThanh/Bidirectional-LSTM-and-Convolutional-Neural-Network-For-Temperature-Prediction/master/data/data.csv')
    st.write(weather_data.head(20))

    st.subheader('Temp distribution on the dataset')
    chart_data = np.histogram(weather_data['Temp'], bins=24, range=(0,24))[0]
    st.bar_chart(chart_data)



with modelTraining:
    st.header('Time to train the model!')
    st.text('Here we get to choose the hyperparameters of the model')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100, 200, 300, 'No limits'])

    sel_col.text('List of input features')
    sel_col.write(weather_data.columns)

    feature_cols = ['precipitation', 'wind_speed','wind_direction','humidity','pressure']
    input_feature = sel_col.selectbox("Input feature", feature_cols)

    if n_estimators == 'No limits':
        regr = RandomForestRegressor(max_depth=max_depth)

    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)


        

    X = weather_data[[input_feature]]
    y = weather_data[['Temp']]

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    regr.fit(X_train, y_train.values.ravel())

    # Prediction
    y_pred = regr.predict(X_test)


    # Evaluation Metrics
    disp_col.subheader('Mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(y_test, y_pred))

    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write(mean_squared_error(y_test, y_pred))

    disp_col.subheader('r2_score of the model is:')
    disp_col.write(r2_score(y_test, y_pred))

