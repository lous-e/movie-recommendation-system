import argparse
from preprocessing import preprocess
import pandas as pd
import time

def parse_arguments():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    
    # Take in arguments for user description, top-n recommendations, and model type
    parser.add_argument('--desc', type=str,required=True,help="User's input to describe preferred movie")
    parser.add_argument('--topn', type=int, default=5, help = "Number of top recommendations to return (default is 5)")
    parser.add_argument('--model', type=str, default='tfidf', choices=['tfidf'], help="Model to use for recommendations (default is tf-idf)")
    
    args = parser.parse_args()
    return args.desc, args.topn, args.model

def main():
    # Start time tracking
    start = time.time()

    print("Starting the movie recommendation system")
    # Parse arguments
    desc, topn, model = parse_arguments()
    topn = max(topn, 10) #ensure that no more than 10 recommendations are made

    # Preprocess data
    preprocess()

    df = pd.read_pickle('cleaned/filtered_df.pkl')
    titles, overviews = df['title'].to_numpy(), df['overview'].to_numpy()

    if model == "tfidf":
        from models.tfidf import TFIDF
        similarities, top_indices = TFIDF(desc, topn, overviews)
        recommendations = df.iloc[top_indices]

        with open("output.txt", "w", encoding="utf-8") as f:
            print(f"\nTop {topn} Recommendations:", file = f)
            for idx, (title, overview, score) in enumerate(zip(recommendations['title'], recommendations['overview'], similarities[top_indices])):
                print(f"{idx+1}. {title} (Similarity: {score:.4f})\n📜 Overview: {overview}\n", file = f)
        
        print(f'Completed {topn} recommendations in {(time.time() - start):.2f} seconds.')
        print('Check output.txt for results.')

if __name__ == '__main__':
    main()