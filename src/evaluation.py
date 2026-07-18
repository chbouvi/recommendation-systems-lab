def precision_at_k(recommended_ids, relevant_ids, k):
    if k == 0:
        return 0
    
    k_recommended = recommended_ids[:k]

    relevant_in_k = len(set(k_recommended).intersection(set(relevant_ids)))

    return relevant_in_k / k

def recall_at_k(recommended_ids, relevant_ids, k):
    k_recommended = recommended_ids[:k]

    if len(relevant_ids) == 0:
        return 0

    relevant_in_k = len(set(k_recommended).intersection(set(relevant_ids)))
    
    return relevant_in_k / len(relevant_ids)

