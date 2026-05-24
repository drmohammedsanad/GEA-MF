import numpy as np


def generate_bpr_samples(R, num_samples=50000):
    """
    Generate training triples for BPR:
    (user, positive item, negative item)

    Parameters:
    - R: user-item interaction matrix
    - num_samples: number of samples to generate

    Returns:
    - list of (u, i, j)
    """

    n_users, n_items = R.shape
    samples = []

    for _ in range(num_samples):
        # Random user
        u = np.random.randint(0, n_users)

        # Positive items (observed interactions)
        pos_items = np.where(R[u] > 0)[0]
        if len(pos_items) == 0:
            continue

        i = np.random.choice(pos_items)

        # Negative items (not interacted)
        neg_items = np.where(R[u] == 0)[0]
        if len(neg_items) == 0:
            continue

        j = np.random.choice(neg_items)

        samples.append((u, i, j))

    return samples


def create_user_groups(R):
    """
    Create user groups based on activity

    Users are divided into:
    - Group 0: low activity
    - Group 1: high activity

    This is used for fairness evaluation
    """

    interaction_counts = np.sum(R, axis=1)

    # Median as threshold
    threshold = np.median(interaction_counts)

    groups = np.zeros(len(interaction_counts), dtype=int)

    for u in range(len(interaction_counts)):
        if interaction_counts[u] < threshold:
            groups[u] = 0
        else:
            groups[u] = 1

    return groups