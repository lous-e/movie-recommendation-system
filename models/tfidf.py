from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def TFIDF(desc, topn, overviews):
  vectorizer = TfidfVectorizer(stop_words='english')
  tfidf_matrix = vectorizer.fit_transform(overviews)
  user_tfidf = vectorizer.transform([desc])
  similarities = cosine_similarity(user_tfidf, tfidf_matrix).reshape(-1)
  top_indices = similarities.argsort()[::-1][:topn]
  return similarities, top_indices