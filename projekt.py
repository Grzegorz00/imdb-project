import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from api_data import APIData
from preprocessing import DataPreprocessor
from modeling import ModelCreator

st.set_page_config(layout="wide")

st.title("Movie Analysis")

api_data = APIData()
df_all = api_data.retrieve_movies_data(num_pages=19)

show_data = st.checkbox("Show data")

if show_data:
    # Display DataFrame
    st.subheader("Movie Data")
    st.dataframe(df_all.head(30))

    # Display DataFrame Statistics
    st.subheader("Data Statistics")
    st.dataframe(df_all.describe())

    # Scatterplot between budget and box office income:
    st.subheader("Scatter Plot: Budget vs Box Office Gross")
    plt.scatter(df_all['budget'], df_all['box_office_gross'], color='green', alpha=0.5)
    plt.xlabel('Budżet')
    plt.ylabel('Dochód z box office')
    st.pyplot(plt)


preprocessor = DataPreprocessor(df_all)
X,Y = preprocessor.preprocess()

model_creator = ModelCreator(X, Y)

# Create and evaluate the models
models, model_results, X_test, y_test = model_creator.create_models()

st.subheader("Model Results")
results_df = pd.DataFrame(model_results).T
results_df = results_df[['mse', 'rmse', 'r2']]
results_df.columns = ['Mean Squared Error', 'Root Mean Squared Error', 'R-squared']
st.table(results_df)

# Display the Ridge model plot
for model_name, results in model_results.items():
    if model_name == 'Ridge':
        y_pred_ridge = results['y_pred']
        model_creator.display_ridge_plot(y_pred_ridge, X_test, y_test)


st.subheader("Predict Movie Rating")
# Create input fields for movie features
release_year = st.number_input("Release Year", min_value=int(df_all['release_year'].min()), max_value=int(df_all['release_year'].max()))
num_votes = st.number_input("Number of Votes", min_value=int(df_all['num_votes'].min()), max_value=int(df_all['num_votes'].max()))
budget = st.number_input("Budget", min_value=int(df_all['budget'].min()), max_value=int(df_all['budget'].max()))
box_office = st.number_input("Box Office Gross", min_value=int(df_all['box_office_gross'].min()), max_value=int(df_all['box_office_gross'].max()))

if st.button("Predict"):
    # Preprocess the input features and reshape
    input_features = np.array([release_year, num_votes, budget, box_office]).reshape(1, -1)

    # Predict the movie rating
    prediction = models[1].predict(input_features)

    # Display the predicted rating
    st.subheader("Movie Rating Prediction")
    st.write("Predicted Rating:", prediction[0])
    st.balloons()