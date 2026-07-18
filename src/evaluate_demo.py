from evaluation import precision_at_k, recall_at_k, hit_rate_at_k

recommended_ids = [1, 2, 3, 4, 5]
relevant_ids = [2, 4, 10]
k = 5

print(f"Precision@{k}: {precision_at_k(recommended_ids, relevant_ids, k)}")
print(f"Recall@{k}: {recall_at_k(recommended_ids, relevant_ids, k)}")
print(f"Hit Rate@{k}: {hit_rate_at_k(recommended_ids, relevant_ids, k)}")
