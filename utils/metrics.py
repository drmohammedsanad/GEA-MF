import numpy as np


def recall_at_k(model, R, K=10):
    """
    Compute Recall@K

    Measures how many relevant items appear in top-K recommendations
    """

    recalls = []

    for u in range(R.shape[0]):
        scores = model.U[u].dot(model.V.T)

        # Get top-K recommended items
        top_k = np.argsort(scores)[-K:]

        # Ground truth (relevant items)
        true_items = np.where(R[u] > 0)[0]

        if len(true_items) == 0:
            continue

        hits = len(set(top_k) & set(true_items))
        recalls.append(hits / len(true_items))

    return np.mean(recalls)


def ndcg_at_k(model, R, K=10):
    """
    Compute NDCG@K

    Considers both relevance AND ranking position
    """

    ndcgs = []

    for u in range(R.shape[0]):
        scores = model.U[u].dot(model.V.T)

        # Sorted recommendations
        top_k = np.argsort(scores)[::-1][:K]

        true_items = np.where(R[u] > 0)[0]

        dcg = 0
        for idx, item in enumerate(top_k):
            if item in true_items:
                dcg += 1 / np.log2(idx + 2)

        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), K)))

        if idcg > 0:
            ndcgs.append(dcg / idcg)

    return np.mean(ndcgs)

def recall_at_k_pop(pop_scores, R, K=10):
    recalls = []

    for u in range(R.shape[0]):
        top_k = np.argsort(pop_scores)[-K:]
        true_items = np.where(R[u] > 0)[0]

        if len(true_items) == 0:
            continue

        hits = len(set(top_k) & set(true_items))
        recalls.append(hits / len(true_items))

    return np.mean(recalls)

def ndcg_at_k_pop(pop_scores, R, K=10):
    import numpy as np

    ndcgs = []

    # Top-K items globally (same for all users)
    top_k = np.argsort(pop_scores)[-K:]

    for u in range(R.shape[0]):
        true_items = np.where(R[u] > 0)[0]

        if len(true_items) == 0:
            continue

        dcg = 0
        for idx, item in enumerate(top_k):
            if item in true_items:
                dcg += 1 / np.log2(idx + 2)

        idcg = sum(
            [1 / np.log2(i + 2) for i in range(min(len(true_items), K))]
        )

        if idcg > 0:
            ndcgs.append(dcg / idcg)

    return np.mean(ndcgs) if len(ndcgs) > 0 else 0
