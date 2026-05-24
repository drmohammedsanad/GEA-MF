import numpy as np
import pandas as pd


# =========================================================
#   MOVIELENS 100K
# =========================================================
def build_movielens_100k(input_path, output_path):
    """
    Convert MovieLens 100K dataset into a binary interaction matrix.

    Input format (u.data):
    userID itemID rating timestamp

    Output:
    - R matrix (users x items)
    - Binary values:
        1 → relevant interaction (rating >= 4)
        0 → no interaction
    """

    # Load dataset using NumPy (space-separated format)
    data = np.loadtxt(input_path, dtype=int)

    # Determine size of matrix
    n_users = data[:, 0].max()
    n_items = data[:, 1].max()

    print(f"[ML-100K] Users: {n_users}, Items: {n_items}")

    # Initialize matrix
    R = np.zeros((n_users, n_items))

    # Fill matrix
    for user, item, rating, _ in data:
        # Convert to implicit feedback (binary)
        if rating >= 4:
            R[user - 1, item - 1] = 1

    # Save matrix
    np.save(output_path, R)

    print(f"  ML-100K matrix saved at {output_path}")
    print(f"  Shape: {R.shape}, Interactions: {np.sum(R)}")


# =========================================================
#   MOVIELENS 1M
# =========================================================
def build_movielens_1m(input_path, output_path):
    """
    Convert MovieLens 1M dataset into a binary interaction matrix.

    Input format (ratings.dat):
    userID::movieID::rating::timestamp

    Notes:
    - Uses pandas because of '::' separator
    - Converts ratings into implicit feedback
    """

    # Load dataset
    data = pd.read_csv(
        input_path,
        sep="::",
        engine="python",
        header=None,
        names=["user", "item", "rating", "timestamp"]
    )

    n_users = data["user"].max()
    n_items = data["item"].max()

    print(f"[ML-1M] Users: {n_users}, Items: {n_items}")

    # Initialize matrix
    R = np.zeros((n_users, n_items))

    # Fill matrix
    for row in data.itertuples():
        if row.rating >= 4:
            R[row.user - 1, row.item - 1] = 1

    # Save matrix
    np.save(output_path, R)

    print(f"  ML-1M matrix saved at {output_path}")
    print(f"  Shape: {R.shape}, Interactions: {np.sum(R)}")


# =========================================================
#   LAST.FM 360K (WITH SAMPLING)
# =========================================================
def build_lastfm(input_path, output_path, min_playcount=5, max_users=5000):
    """
    Convert Last.fm dataset into a binary interaction matrix.

    Input format:
    userID\tartistID\tplaycount

    Steps:
    1. Load data
    2. Sample subset of users (IMPORTANT for memory)
    3. Map IDs to indices
    4. Build binary matrix

    Parameters:
    - min_playcount: threshold to define interaction
    - max_users: number of users to include (CRITICAL for scalability)
    """

    # -----------------------------
    # STEP 1: Load dataset
    # -----------------------------
    data = pd.read_csv(
        input_path,
        sep="\t",
        header=None,
        names=["user", "item", "playcount"]
    )

    print(f"[LastFM] Original data shape: {data.shape}")

    # -----------------------------
    # STEP 2: USER SAMPLING (VERY IMPORTANT)
    # -----------------------------
    # Get unique users
    unique_users = data["user"].unique()

    # Select subset (first N users)
    sampled_users = unique_users[:max_users]

    # Filter dataset
    data = data[data["user"].isin(sampled_users)]

    print(f"[LastFM] After sampling {max_users} users: {data.shape}")

    # -----------------------------
    # STEP 3: Create ID mappings
    # -----------------------------
    user_ids = data["user"].unique()
    item_ids = data["item"].unique()

    # Map original IDs → continuous indices
    user_map = {u: idx for idx, u in enumerate(user_ids)}
    item_map = {i: idx for idx, i in enumerate(item_ids)}

    n_users = len(user_ids)
    n_items = len(item_ids)

    print(f"[LastFM] Final users: {n_users}, items: {n_items}")

    # -----------------------------
    # STEP 4: Build matrix
    # -----------------------------
    R = np.zeros((n_users, n_items))

    for row in data.itertuples():
        if row.playcount >= min_playcount:
            u = user_map[row.user]
            i = item_map[row.item]
            R[u, i] = 1

    # -----------------------------
    # STEP 5: Save matrix
    # -----------------------------
    np.save(output_path, R)

    print(f"  LastFM matrix saved at {output_path}")
    print(f"  Shape: {R.shape}, Interactions: {np.sum(R)}")


# =========================================================
#   MAIN EXECUTION
# =========================================================
if __name__ == "__main__":

    """
    IMPORTANT:
    Uncomment ONLY ONE dataset at a time
    """

    # -----------------------------
    #   MOVIELENS 100K
    # -----------------------------
    build_movielens_100k(
         input_path="../data/raw/ml-100k/u.data",
         output_path="../data/processed/R_100k.npy"
     )

    # -----------------------------
    #   MOVIELENS 1M
    # -----------------------------
    #build_movielens_1m(
    #     input_path="../data/raw/ml-1m/ratings.dat",
    #     output_path="../data/processed/R_1m.npy"
    # )

    # -----------------------------
    #   LAST.FM 360K
    # -----------------------------
    #build_lastfm(
    #    input_path="../data/raw/lastfm-360k/usersha1-artmbid-artname-plays.tsv",
    #    output_path="../data/processed/R_lastfm.npy",
    #    min_playcount=5,
    #    max_users=20000
    #)
