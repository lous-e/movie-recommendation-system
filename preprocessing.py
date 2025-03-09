import pandas as pd
import numpy as np
import os


def preprocess():
    # Load the dataset, filter out movies without fields, remove duplicates
    df = pd.read_csv("data/tmdb_5000_movies.csv")  # shape: 4803 x 20
    filtered_df = df[
        df[["title", "overview"]].notnull().all(axis=1)
    ]  # shape: 4800 x 20
    filtered_df = filtered_df.drop_duplicates(subset="title")  # shape: 4797 x 20
    print(f"Preprocessed {len(filtered_df)} movies")

    # Pickle the filtered DataFrame
    if not os.path.exists("cleaned"):
        os.makedirs("cleaned")
    filtered_df.to_pickle("cleaned/filtered_df.pkl")
    print("Data saved to cleaned/filtered_df")
