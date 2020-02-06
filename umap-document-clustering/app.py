import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from umap import UMAP

from yellowbrick.datasets import load_hobbies
from yellowbrick.text import UMAPVisualizer

"""
# UMAP for clustering
We will cluster some text documents (the
[hobbies corpus](https://www.scikit-yb.org/en/latest/api/datasets/hobbies.html))
based on their tf-idf representation, and compare that to clustering on the umap
embedding of their tf-idf representation.
"""

@st.cache(allow_output_mutation=True)
def load_document_representation():
    corpus = load_hobbies()
    tfidf = TfidfVectorizer()
    docs = tfidf.fit_transform(corpus.data)
    return docs, corpus.target

docs, labels = load_document_representation()


"""
First we cluster on the tf-idf embedding directly.
We keep things simple with k-means, but note that requires choosing a number
of clusters.
"""

tfidf_labels = [
    "cluster {}".format(label)
    for label in
    KMeans(n_clusters=5).fit(docs).labels_
]

tfidf_umap = UMAPVisualizer(metric='cosine').fit(docs, tfidf_labels)

st.pyplot(tfidf_umap._fig)


"""
Ok, looks like the clusters don't map especially well to the umap embedding.
Let's try first umap-ing, then clustering.
"""

embeddings = UMAP(metric='cosine').fit_transform(docs)

umap_labels = [
    "cluster {}".format(label)
    for label in
    KMeans(n_clusters=5).fit(embeddings).labels_
]

umap_viz = UMAPVisualizer(metric='cosine').fit(docs, umap_labels)

st.pyplot(umap_viz._fig)


"""
Naturally, the clusters are more separated in the UMAP space, since that's
where they were clustered!
(Actually, it's not perfect, since different instances of UMAP were used for
the initial embedding (to cluster) and the viz. This is super alpha.)

## So, which is the better set of embeddings?

Let's just visually compare with the true labels.
"""

true_viz = UMAPVisualizer(metric='cosine').fit(docs, labels)

st.pyplot(true_viz._fig)


"""
To my eye, the UMAP embeddings look closer! We could easily check this with
by treating the clustering as classification (since we have a true label), but
this is just a silly little POC, afterall.
Also, out there in the wild in a clustering problem, we're unlikely to have
labels, so that it works as a classification method is really just a bonus
aside.
"""