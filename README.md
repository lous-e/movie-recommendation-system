# Movie Recommendation System

## Overview

Coming Soon!

## Folder Structure

Coming Soon!

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
#### Mac
```{bash}
python3 -m venv venv
source ./venv/bin/activate
```
### Install dependencies
```{bash}
pip install -r requirements.txt
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

### Outputs

The recommendations are saved in outputs/{out}.txt with details including movie title, similarity score, and overview.

## Results

## Link to screen recording
## Future Work