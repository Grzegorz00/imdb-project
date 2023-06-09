from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

class ModelCreator:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def create_models(self):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Create a list of models
        models = [
            LinearRegression(),
            Ridge(),
            Lasso(),
            RandomForestRegressor(),
            SVR()
        ]

        # Prepare a dictionary for model results
        model_results = {}

        # Iterate over the models
        for model in models:
            model_name = model.__class__.__name__

            # Create the model
            model.fit(X_train, y_train)

            # Make predictions on the test data
            y_pred = model.predict(X_test)

            # Calculate model evaluation metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = model.score(X_test, y_test)

            # Save the model results
            model_results[model_name] = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'y_pred': y_pred
            }

        return models, model_results, X_test, y_test
    
    def display_ridge_plot(self, y_pred_ridge, X_test, y_test):
        # Create the scatter plot for Ridge model predictions vs. actual values
        fig, ax = plt.subplots()
        ax.scatter(X_test['num_votes'], y_test, color='blue', label='Actual grades')
        ax.scatter(X_test['num_votes'], y_pred_ridge, color='red', label='Ratings provided (Ridge)')
        ax.set_xlabel('Number of votes')
        ax.set_ylabel('Average rating')
        ax.set_title('Prediction vs actual values (Ridge)')
        ax.legend()
        st.pyplot(fig)
