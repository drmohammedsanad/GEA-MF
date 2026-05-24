import numpy as np


def compute_exposure(model, R, K=10):
    """
    Compute item exposure based on Top-K recommendations
    """

    n_users = R.shape[0]
    n_items = R.shape[1]

    exposure = np.ones(n_items)  # initialize with 1 to avoid division by zero

    for u in range(n_users):
        scores = model.U[u].dot(model.V.T)
        top_k = np.argsort(scores)[-K:]

        for item in top_k:
            exposure[item] += 1

    return exposure


def compute_exposure_weights(exposure, epsilon=1e-6):
    """
    Compute inverse exposure weights

    Higher exposure → lower weight
    """

    weights = 1.0 / (exposure + epsilon)

    # Normalize weights for stability
    weights = weights / np.mean(weights)

    return weights


def exposure_parity(model, R, item_groups, K=10):
    """
    Compute fairness: difference in exposure between groups
    """

    exposure = compute_exposure(model, R, K)

    group_0 = np.where(item_groups == 0)[0]
    group_1 = np.where(item_groups == 1)[0]

    exp_0 = np.mean(exposure[group_0])
    exp_1 = np.mean(exposure[group_1])

    return abs(exp_0 - exp_1)


def compute_item_groups(n_items):
    """
    Split items into 2 groups (integer labels)
    """

    groups = np.zeros(n_items, dtype=int)  # ✅ MUST BE INT
    groups[n_items // 2:] = 1

    return groups

def exposure_parity_pop(pop_scores, R, item_groups, K=10):
    import numpy as np

    n_items = len(pop_scores)
    exposure = np.zeros(n_items)

    # Top-K items globally
    top_k = np.argsort(pop_scores)[-K:]

    for u in range(R.shape[0]):
        for item in top_k:
            exposure[item] += 1

    group_0 = np.where(item_groups == 0)[0]
    group_1 = np.where(item_groups == 1)[0]

    exp_0 = np.mean(exposure[group_0])
    exp_1 = np.mean(exposure[group_1])

    return abs(exp_0 - exp_1)