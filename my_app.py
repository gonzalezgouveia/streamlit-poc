import streamlit as st
import pandas as pd
import numpy as np
import pickle
# from sklearn.ensemble import RandomForestClassifier

st.write("""
# Price Prediction App
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file]
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        # island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        # sex = st.sidebar.selectbox('Sex',('male','female'))
        LotArea = st.sidebar.slider('LotArea', 1000,250000,100000)
        MSSubClass = st.sidebar.slider('MSSubClass', 20,190,100)
        OverallCond = st.sidebar.slider('OverallCond', 1,9,5)
        YearBuilt = st.sidebar.slider('YearBuilt', 1800,2100,2000)
        data = {
                #'island': island,
                'LotArea': LotArea,
                'MSSubClass': MSSubClass,
                'OverallCond': OverallCond,
                'YearBuilt': YearBuilt,
                #'sex': sex
        }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# # Combines user input features with entire penguins dataset
# # This will be useful for the encoding phase
# penguins_raw = pd.read_csv('penguins_cleaned.csv')
# penguins = penguins_raw.drop(columns=['species'])
# df = pd.concat([input_df,penguins],axis=0)

# # Encoding of ordinal features
# # https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# encode = ['sex','island']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]
# df = df[:1] # Selects only the first row (the user input data)

df = input_df

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('linreg.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
# prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
# penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(prediction)

# st.subheader('Prediction Probability')
# st.write(prediction_proba)