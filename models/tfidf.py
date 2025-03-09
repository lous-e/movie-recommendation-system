from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def recommend(desc, topn, overviews):
    # create and fit the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(overviews)

    # get vector for user description
    user_tfidf = vectorizer.transform([desc])

    # calculate cosine similarity
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).reshape(-1)

    # get top n indices
    top_indices = similarities.argsort()[::-1][:topn]
    return similarities, top_indices
