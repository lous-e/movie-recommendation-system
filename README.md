# Movie Recommendation System

## Overview

This project implements a movie recommendation system using various NLP techniques to match user preferences with movie descriptions. TF-IDF, lemmatized TF-IDF, SVD-reduced TF-IDF, and SBERT embeddings to provide content recommendations based on user inputs.

## Folder Structure
```
movie-recommendation-system/
├── cleaned/                  # Preprocessed data files (ignored in git)
│   └── filtered_df.pkl      # Filtered dataset
├── data/                    # Raw data 
│   └── tmdb_5000_movies.csv
├── models/                  # Model implementations
│   ├── sbert.py            # Sentence-BERT model
│   ├── tfidf.py            # Basic TF-IDF model
│   ├── tfidf_lemmatized.py # Lemmatized TF-IDF
│   └── tfidf_svd.py        # TF-IDF with SVD
├── outputs/                 # Recommendation outputs (ignored in git)
├── client.py               # CLI interface
├── preprocessing.py        # Data preprocessing scripts
├── README.md        # this file
├── demo.md        # link to video demo
└── requirements.txt        # Project dependencies
```

## Dataset
We used a publically available dataset from Kaggle called the [TMDb 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv). This is a list of around 5k popular movies with plot overviews and other related data collected around 7 years ago. The database was generated using the [TMBd](https://www.themoviedb.org/) API.

## Setup
### Clone and navigate to root
```{bash}
git clone https://github.com/lous-e/movie-recommendation-system
cd movie-recommendation-system
```


### Install dependencies
Create a new virtual environment using any tool you prefer. We use venv for this example

#### Windows
```{bash}
python -m venv venv
venv/Scripts/activate
```
#### MacOS
```{bash}
python3 -m venv venv
source ./venv/bin/activate
```
### Install dependencies
```{bash}
pip install -r requirements.txt
python load_models.py
```

## Usage
### CLI Tool
```{bash}
python client.py --desc "I like action movies set in space" --topn 5 --model tfidf --out recommendations
```
### Arguments
| Argument        | Description           | Type  | Default
| ------------- |:-------------| :-----:| -----:
| --desc      | User input describing preference | str | Required
| --topn      | Number of top recommendations to return      |   int | 5 (Max 10)
| --model | Model type for recommendations      |   str | tfidf
| --out | Output file name (saved in outputs/)      |    str | output

Currently, the following models are supported:
- ```tfidf```: Returns the top-n movies sorted by descending order of [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).
- ```tfidf-lemmatized```: [Lemmatizes](https://en.wikipedia.org/wiki/Lemmatization) the words before tf-idf.
- ```tfidf-svd```: Performs [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) on tfidf matrices to reduce dimensionality.
- ```sbert```: Uses SBERT embeddings for semantic similarity matching

### Outputs
The recommendations are saved in outputs/{out}.txt with details including movie title, similarity score, and overview.

## Results
Top-5 movie recommendations for sample query

```
I like space adventure films
```

- tfidf
    1. The Kentucky Fried Movie (Similarity: 0.4654)
    2. Space Pirate Captain Harlock (Similarity: 0.2361)
    3. A Haunted House (Similarity: 0.2192)
    4. Metallica: Through the Never (Similarity: 0.1830)
    5. Lifeforce (Similarity: 0.1696)

- tfidf-lemmatized
    1. The Kentucky Fried Movie (Similarity: 0.4654)
    2. Space Pirate Captain Harlock (Similarity: 0.2361)
    3. A Haunted House (Similarity: 0.2192)
    4. Metallica: Through the Never (Similarity: 0.1830)
    5. Lifeforce (Similarity: 0.1696)

- tfidf-svd
    1. Lost in Space (Similarity: 0.4337)
    2. Space Pirate Captain Harlock (Similarity: 0.4144)
    3. Moonraker (Similarity: 0.3778)
    4. Deck the Halls (Similarity: 0.3716)
    5. The Kentucky Fried Movie (Similarity: 0.3600)

- sbert
    1. Interstellar (Similarity: 0.4547)
    2. You Only Live Twice (Similarity: 0.4534)
    3. Sea Rex 3D: Journey to a Prehistoric World (Similarity: 0.4388)
    4. My Big Fat Independent Movie (Similarity: 0.4096)
    5. Galaxy Quest (Similarity: 0.4080)

## Future Work
- Model Improvements
    - Add collaborative filtering based on user ratings
    - Incorporate more advanced transformer models
- Real-time movie data updates
- Deployment
    - Dockerization
    - Endpoints using FastAPI
    - Streamlit frontend
- Evaluation

## Expectations
Commented in PR!