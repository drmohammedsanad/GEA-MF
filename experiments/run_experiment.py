import os
import numpy as np
from datetime import datetime
from collections import defaultdict

from models.fair_ranking_mf import FairRankingMF
from models.mf import MF
from models.bpr import BPR
from models.svdpp import SVDpp
from models.popularity import PopularityModel

from utils.data_loader import load_data
from utils.sampler import generate_bpr_samples
from utils.metrics import (
    recall_at_k,
    ndcg_at_k,
    recall_at_k_pop,
    ndcg_at_k_pop
)
from utils.fairness import (
    exposure_parity,
    exposure_parity_pop,
    compute_item_groups
)

from experiments.config import config


# ===================================================
# ✅ LOAD DATASET
# ===================================================
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


# ===================================================
# ✅ RUN SINGLE DATASET (WITH MULTIPLE RUNS ✅)
# ===================================================
def run_single_dataset(dataset_name, n_runs=5):

    print("\n===================================")
    print(f"🚀 Dataset: {dataset_name}")
    print("===================================")

    R = load_dataset(dataset_name)
    item_groups = compute_item_groups(R.shape[1])

    all_runs = defaultdict(list)

    # ----------------------------------------
    # ✅ MULTIPLE RUNS
    # ----------------------------------------
    for run in range(n_runs):

        print(f"\n================ RUN {run+1}/{n_runs} ================\n")

        samples = generate_bpr_samples(R, config["samples"])
        results = {}

        # --------------------------------------
        # ✅ POPULARITY
        # --------------------------------------
        pop = PopularityModel()
        pop.train(R)
        pop_scores = pop.predict_scores()

        results["Popularity"] = {
            "recall": recall_at_k_pop(pop_scores, R),
            "ndcg": ndcg_at_k_pop(pop_scores, R),
            "parity": exposure_parity_pop(pop_scores, R, item_groups)
        }

        # --------------------------------------
        # ✅ MF
        # --------------------------------------
        mf = MF(R.shape[0], R.shape[1])
        mf.train(R, config["epochs"])

        results["MF"] = {
            "recall": recall_at_k(mf, R),
            "ndcg": ndcg_at_k(mf, R),
            "parity": exposure_parity(mf, R, item_groups)
        }

        # --------------------------------------
        # ✅ SVD++
        # --------------------------------------
        svdpp = SVDpp(R.shape[0], R.shape[1])
        svdpp.train(R, config["epochs"])

        results["SVD++"] = {
            "recall": recall_at_k(svdpp, R),
            "ndcg": ndcg_at_k(svdpp, R),
            "parity": exposure_parity(svdpp, R, item_groups)
        }

        # --------------------------------------
        # ✅ BPR
        # --------------------------------------
        bpr = BPR(R.shape[0], R.shape[1])
        bpr.train(samples, config["epochs"])

        results["BPR"] = {
            "recall": recall_at_k(bpr, R),
            "ndcg": ndcg_at_k(bpr, R),
            "parity": exposure_parity(bpr, R, item_groups)
        }

        # --------------------------------------
        # ✅ FR-MF (FINAL MODEL)
        # --------------------------------------
        frmf = FairRankingMF(
            R.shape[0],
            R.shape[1],
            k=config["k"],
            lr=config["lr"],
            reg=config["reg"],
            alpha=config["alpha"]
        )

        frmf.item_groups = item_groups

        frmf.train(
            samples,
            R,
            n_epochs=config["epochs"],
            top_k=config["top_k"]
        )

        results["FR-MF"] = {
            "recall": recall_at_k(frmf, R),
            "ndcg": ndcg_at_k(frmf, R),
            "parity": exposure_parity(frmf, R, item_groups)
        }

        # --------------------------------------
        # ✅ STORE RUN RESULTS
        # --------------------------------------
        for model, m in results.items():
            all_runs[model].append(m)

    # ----------------------------------------
    # ✅ COMPUTE MEAN ± STD
    # ----------------------------------------
    final_results = {}

    for model, runs in all_runs.items():

        recall = [r["recall"] for r in runs]
        ndcg = [r["ndcg"] for r in runs]
        parity = [r["parity"] for r in runs]

        final_results[model] = {
            "recall_mean": np.mean(recall),
            "recall_std": np.std(recall),

            "ndcg_mean": np.mean(ndcg),
            "ndcg_std": np.std(ndcg),

            "parity_mean": np.mean(parity),
            "parity_std": np.std(parity),
        }

    # ----------------------------------------
    # ✅ PRINT FINAL TABLE
    # ----------------------------------------
    print("\n📊 FINAL RESULTS TABLE (mean ± std)\n")

    for model, m in final_results.items():
        print(f"{model:12} | "
              f"Recall={m['recall_mean']:.4f}±{m['recall_std']:.4f} | "
              f"NDCG={m['ndcg_mean']:.4f}±{m['ndcg_std']:.4f} | "
              f"Exposure={m['parity_mean']:.4f}±{m['parity_std']:.4f}")

    return final_results


# ===================================================
# ✅ SAVE RESULTS
# ===================================================
def save_results(all_results):

    os.makedirs("results/logs/", exist_ok=True)

    filename = f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path = os.path.join("results/logs/", filename)

    with open(path, "w") as f:

        f.write("=== FINAL RESULTS (mean ± std) ===\n\n")

        for dataset, results in all_results.items():

            f.write(f"\nDataset: {dataset}\n")
            f.write("-----------------------------------\n")

            for model, m in results.items():
                f.write(
                    f"{model:12} | "
                    f"Recall={m['recall_mean']:.4f}±{m['recall_std']:.4f} | "
                    f"NDCG={m['ndcg_mean']:.4f}±{m['ndcg_std']:.4f} | "
                    f"Exposure={m['parity_mean']:.4f}±{m['parity_std']:.4f}\n"
                )

    print(f"\n✅ Results saved to {path}")


# ===================================================
# ✅ MAIN RUN
# ===================================================
def run():

    # ✅ Start with ml-100k; later extend
    datasets = ["ml-100k", "ml-1m", "lastfm"]

    #datasets = ["ml-100k"]
    #datasets = ["ml-1m"]

    all_results = {}

    for d in datasets:
        all_results[d] = run_single_dataset(d, n_runs=5)

    save_results(all_results)
