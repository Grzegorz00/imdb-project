import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def fill_missing_values(self):
        mean_budget = self.df['budget'].mean()
        self.df['budget'].fillna(mean_budget, inplace=True)

        mean_box_office = self.df['box_office_gross'].mean()
        self.df['box_office_gross'].fillna(mean_box_office, inplace=True)

    def filter_dataframe(self, max_budget_threshold):
        self.df = self.df[self.df['budget'] <= max_budget_threshold]

    def drop_missing_values(self):
        self.df = self.df.dropna(subset=['id', 'title', 'genres', 'average_rating', 'num_votes', 'runtimes', 'release_year'])

    def encode_genres(self):
        encoded_genres = pd.get_dummies(self.df['genres'], prefix='', prefix_sep='')
        self.df = pd.concat([self.df, encoded_genres], axis=1)
        self.df.drop('genres', axis=1, inplace=True)

    def make_plots(self):
        plot_type = st.selectbox("Select Plot Type", ['Histograms', 'Relationship between release date and rating', 'Relationship between budget and  box office', 'Relationship between runtimes and rating'])

        if plot_type == 'Histograms':
            st.subheader("Histograms")
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            columns = ['average_rating', 'num_votes', 'runtimes', 'budget', 'box_office_gross']

            for ax, column in zip(axes, columns):
                ax.hist(self.df[column], bins=10)
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')

            plt.tight_layout()
            st.pyplot(fig)

        elif plot_type == 'Relationship between release date and rating':
            st.subheader("Relationship between release date and rating")
            average_rating_by_year = self.df.groupby('release_year')['average_rating'].mean()
            fig, ax = plt.subplots()
            ax.plot(average_rating_by_year.index, average_rating_by_year.values, color='orange')
            ax.set_xlabel('Year')
            ax.set_ylabel('Avarage rating')
            st.pyplot(fig)

        elif plot_type == 'Relationship between budget and box office':
            st.subheader("Relationship between budget and box office")
            fig, ax = plt.subplots()
            ax.scatter(self.df['budget'], self.df['box_office_gross'], color='green', alpha=0.5)
            ax.set_xlabel('Budget')
            ax.set_ylabel('Box Office Gross')
            st.pyplot(fig)

        elif plot_type == 'Relationship between runtimes and rating':
            st.subheader("Relationship between runtimes and rating")
            fig, ax = plt.subplots()
            ax.scatter(self.df['runtimes'], self.df['average_rating'], color='green', alpha=0.5)
            ax.set_xlabel('Runtimes')
            ax.set_ylabel('Avarage rating')
            st.pyplot(fig)

    def get_features_and_target(self):
        X = self.df[['release_year', 'num_votes', 'budget', 'box_office_gross']]
        y = self.df['average_rating']
        return X, y

    def preprocess(self):
        self.fill_missing_values()
        self.filter_dataframe(40000000)
        self.make_plots()
        self.drop_missing_values()
        self.encode_genres()
        return self.get_features_and_target()