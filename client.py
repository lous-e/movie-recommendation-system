import argparse
from preprocessing import preprocess
import pandas as pd
import time
import os

def parse_arguments():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    
    # Take in arguments for user description, top-n recommendations, and model type
    parser.add_argument('--desc', type=str,required=True,help="User input describing preference")
    parser.add_argument('--topn', type=int, default=5, help = "Number of top recommendations to return")
    parser.add_argument('--model', type=str, default='tfidf', choices=['tfidf', 'tfidf-lemmatized', 'tfidf-svd'], help="Model type for recommendations")
    parser.add_argument('--out', type=str, default='output', help="Output file name (saved in outputs/)")
    
    args = parser.parse_args()
    return args.desc, args.topn, args.model, args.out

def main():
    # Start time tracking
    start = time.time()

    print("Starting the movie recommendation system...")

    # Parse arguments
    desc, topn, model, out = parse_arguments()
    topn = max(topn, 10) if topn >= 10 else topn #ensure that no more than 10 recommendations are made

    # Preprocess data
    preprocess()

    # Load the prep
    df = pd.read_pickle('cleaned/filtered_df.pkl')
    overviews = df['overview'].to_numpy()

    # Load relevant model

    # base tfidf model
    if model == "tfidf":
        from models.tfidf import recommend
    # tfidf with lemmatization
    elif model == "tfidf-lemmatized":
        from models.tfidf_lemmatized import recommend
    # tfidf with svd
    elif model == "tfidf-svd":
        from models.tfidf_svd import recommend
    
    # get recommendations
    similarities, top_indices = recommend(desc, topn, overviews)
    recommendations = df.iloc[top_indices]

    # create output folder
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # write to output file
    output_file = f"outputs/{out}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        print(f"\nTop {topn} Recommendations:", file = f)
        for idx, (title, overview, score) in enumerate(zip(recommendations['title'], recommendations['overview'], similarities[top_indices])):
            print(f"{idx+1}. {title} (Similarity: {score:.4f})\n📜 Overview: {overview}\n", file = f)

    # print completion message
    print(f'Completed {topn} recommendations in {(time.time() - start):.2f} seconds.')
    print(f'Check outputs/{out}.txt for results.')

if __name__ == '__main__':
    main()