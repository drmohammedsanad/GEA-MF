import os
from datetime import datetime

from models.fair_ranking_mf import FairRankingMF
from models.mf import MF
from models.bpr import BPR
from models.svdpp import SVDpp
from models.popularity import PopularityModel

from utils.data_loader import load_data
from utils.sampler import generate_bpr_samples
from utils.metrics import recall_at_k, ndcg_at_k, recall_at_k_pop
from utils.fairness import (
    compute_exposure,
    compute_exposure_weights,
    compute_item_groups,
    exposure_parity
)

from experiments.config import config


# ========================================
# ✅ Load dataset
# ========================================
def load_dataset(name):

    base = "data/processed/"

    paths = {
        "ml-100k": "R_100k.npy",
        "ml-1m": "R_1m.npy",
        "lastfm": "R_lastfm.npy"
    }

    path = os.path.join(base, paths[name])

    print(f"\n📂 Loading dataset: {name}")
    R = load_data(path)

    print(f"✅ Shape: {R.shape}, Interactions: {R.sum()}")
    return R


# ========================================
# ✅ Run single dataset
# ========================================
def run_single_dataset(dataset_name):

    print("\n===================================")
    print(f"🚀 Dataset: {dataset_name}")
    print("===================================")

    R = load_dataset(dataset_name)
    samples = generate_bpr_samples(R, config["samples"])

    item_groups = compute_item_groups(R.shape[1])
    results = {}

    # --------------------------------------------------
    # ✅ Popularity
    # --------------------------------------------------
    print("\n🔹 Training Popularity...")
    pop = PopularityModel()
    pop.train(R)
    pop_scores = pop.predict_scores()

    results["Popularity"] = {
        "recall": recall_at_k_pop(pop_scores, R),
        "ndcg": 0.0,
        "parity": 0.0
    }

    # --------------------------------------------------
    # ✅ MF
    # --------------------------------------------------
    print("\n🔹 Training MF...")
    mf = MF(R.shape[0], R.shape[1])
    mf.train(R, config["epochs"])

    results["MF"] = {
        "recall": recall_at_k(mf, R),
        "ndcg": ndcg_at_k(mf, R),
        "parity": exposure_parity(mf, R, item_groups)
    }

    # --------------------------------------------------
    # ✅ SVD++
    # --------------------------------------------------
    print("\n🔹 Training SVD++...")
    svdpp = SVDpp(R.shape[0], R.shape[1])
    svdpp.train(R, config["epochs"])

    results["SVD++"] = {
        "recall": recall_at_k(svdpp, R),
        "ndcg": ndcg_at_k(svdpp, R),
        "parity": exposure_parity(svdpp, R, item_groups)
    }

    # --------------------------------------------------
    # ✅ BPR (IMPORTANT BASELINE)
    # --------------------------------------------------
    print("\n🔹 Training BPR...")
    bpr = BPR(R.shape[0], R.shape[1])
    bpr.train(samples, config["epochs"])

    results["BPR"] = {
        "recall": recall_at_k(bpr, R),
        "ndcg": ndcg_at_k(bpr, R),
        "parity": exposure_parity(bpr, R, item_groups)
    }

    # --------------------------------------------------
    # ✅ FR-MF (ALPHA SWEEP ✅)
    # --------------------------------------------------

    alphas = config["alpha_list"]

    for alpha in alphas:
        print(f"\n🔥 Training FR-MF with alpha = {alpha}...")

        frmf = FairRankingMF(
            R.shape[0],
            R.shape[1],
            k=config["k"],
            lr=config["lr"],
            reg=config["reg"],
            alpha=alpha  # ✅ dynamic alpha
        )

        frmf.item_groups = item_groups

        frmf.train(
            samples,
            R,
            n_epochs=config["epochs"],
            top_k=config["top_k"]
        )

        # ✅ Save result with alpha label
        model_name = f"FR-MF (α={alpha})"

        results[model_name] = {
            "recall": recall_at_k(frmf, R),
            "ndcg": ndcg_at_k(frmf, R),
            "parity": exposure_parity(frmf, R, item_groups)
        }

        print(f"{model_name}: "
              f"Recall={results[model_name]['recall']:.4f}, "
              f"NDCG={results[model_name]['ndcg']:.4f}, "
              f"Exposure={results[model_name]['parity']:.4f}")

    # --------------------------------------------------
    # ✅ PRINT RESULTS
    # --------------------------------------------------
    print("\n📊 RESULTS TABLE")

    for model, m in results.items():
        print(f"{model:12} | Recall={m['recall']:.4f} "
              f"| NDCG={m['ndcg']:.4f} "
              f"| Exposure={m['parity']:.4f}")

    return results


# ========================================
# ✅ Save results
# ========================================
def save_results(all_results):

    os.makedirs("results/logs/", exist_ok=True)

    filename = f"full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path = os.path.join("results/logs/", filename)

    with open(path, "w") as f:

        f.write("=== FINAL EXPERIMENT RESULTS ===\n\n")

        for dataset, results in all_results.items():

            f.write(f"\nDataset: {dataset}\n")
            f.write("-----------------------------------\n")

            for model, m in results.items():
                f.write(
                    f"{model:12} | "
                    f"Recall={m['recall']:.4f} | "
                    f"NDCG={m['ndcg']:.4f} | "
                    f"Exposure={m['parity']:.4f}\n"
                )

    print(f"\n✅ Results saved to {path}")


# ========================================
# ✅ MAIN
# ========================================
def run():

    #datasets = ["ml-100k", "ml-1m", "lastfm"]
    datasets = ["ml-100k"]

    all_results = {}

    for d in datasets:
        all_results[d] = run_single_dataset(d)

    save_results(all_results)