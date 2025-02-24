from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

def recommend(desc, topn, overviews):
  # create and fit the TF-IDF vectorizer
  vectorizer = TfidfVectorizer(stop_words='english')
  tfidf_matrix = vectorizer.fit_transform(overviews)

  # get vector for user description
  user_tfidf = vectorizer.transform([desc])

  # perform reduced dimensionality SVD
  svd = TruncatedSVD(n_components=500, random_state=42)
  tfidf_matrix_svd = csr_matrix(svd.fit_transform(tfidf_matrix))
  user_svd = csr_matrix(svd.transform(user_tfidf))

  # calculate cosine similarity
  similarities = cosine_similarity(user_svd, tfidf_matrix_svd).reshape(-1)

  # get top n indices
  top_indices = similarities.argsort()[::-1][:topn]
  return similarities, top_indices