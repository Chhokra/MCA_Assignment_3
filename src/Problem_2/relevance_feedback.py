import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    non_sparse_vec_docs = vec_docs.toarray()
    non_sparse_vec_queries = vec_queries.toarray()
    for z in range(3):

        for i in range(30):
            relevant_documents = np.argsort(-sim[:,i])[:n]
            relevant_lis = list(relevant_documents)
            non_relevant_lis = []
            for j in range(1033):
                if(j not in relevant_lis):
                    non_relevant_lis.append(j)
            sum1 = np.zeros(10625)
            for j in relevant_lis:
                sum1 = sum1 + non_sparse_vec_docs[j]
            sum2 = np.zeros(10625)
            for j in non_relevant_lis:
                sum2 = sum2 + non_sparse_vec_docs[j]
            non_sparse_vec_queries[i] = non_sparse_vec_queries[i] + (1/10)*sum1 - (1/1023)*sum2 

        sim = cosine_similarity(non_sparse_vec_docs,non_sparse_vec_queries)


    rf_sim = sim # change
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    non_sparse_vec_docs = vec_docs.toarray()
    non_sparse_vec_queries = vec_queries.toarray()
    for z in range(3):

        for i in range(30):
            relevant_documents = np.argsort(-sim[:,i])[:n]
            relevant_lis = list(relevant_documents)
            non_relevant_lis = []
            for j in range(1033):
                if(j not in relevant_lis):
                    non_relevant_lis.append(j)
            sum1 = np.zeros(10625)
            for j in relevant_lis:
                sum1 = sum1 + non_sparse_vec_docs[j]
            sum2 = np.zeros(10625)
            for j in non_relevant_lis:
                sum2 = sum2 + non_sparse_vec_docs[j]
            non_sparse_vec_queries[i] = non_sparse_vec_queries[i] + (1/10)*sum1 - (1/1023)*sum2 

        sim = cosine_similarity(non_sparse_vec_docs,non_sparse_vec_queries)
    normalizer = preprocessing.Normalizer().fit(non_sparse_vec_docs)
    non_sparse_vec_docs = normalizer.transform(non_sparse_vec_docs)

    term_similarity_matrix = np.matmul(non_sparse_vec_docs.T,non_sparse_vec_docs)

    for i in range(30):
        query = non_sparse_vec_queries[i]
        max_index = np.argmax(query)
        most_important_terms = np.argsort(-term_similarity_matrix[:,max_index])[:8]
        for j in most_important_terms:
            non_sparse_vec_queries[i][j] = query[max_index]
    sim_new = cosine_similarity(non_sparse_vec_docs,non_sparse_vec_queries)



    rf_sim = sim_new  # change
    return rf_sim