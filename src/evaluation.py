def precision_at_k(recommended_ids, relevant_ids, k):
    k_recommended = recommended_ids[:k]

    relevant_in_k = len(set(k_recommended).intersection(set(relevant_ids)))

    return relevant_in_k / k



