from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def recommend(desc, topn, overviews):
    # Load the model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get embeddings for the movie overviews
    movie_embeddings = model.encode(overviews)

    # Get embedding for the user description
    user_embedding = model.encode([desc])

    # Calculate cosine similarity
    similarities = cosine_similarity(user_embedding, movie_embeddings).reshape(-1)

    # Get top n indices
    top_indices = similarities.argsort()[::-1][:topn]

    return similarities, top_indices
