from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy


def recommend(desc, topn, overviews):
    # preprocess (lemmatize)
    nlp = spacy.load(
        "en_core_web_sm", enable=["tagger", "attribute_ruler", "lemmatizer"]
    )
    lemmatize = lambda text: [
        " ".join([token.lemma_ for token in doc if not token.is_stop])
        for doc in nlp.pipe(text)
    ]
    overviews = lemmatize(overviews)
    desc = lemmatize([desc])[0]

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
