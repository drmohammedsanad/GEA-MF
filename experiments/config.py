config = {
    # ✅ Choose dataset here
    #"dataset": "ml-100k",   # ← change to ml-1m / lastfm
    #"dataset": "ml-1m",  # ← change to ml-1m / lastfm
    #"dataset": "lastfm",  # ← change to ml-1m / lastfm

    # Model parameters
    "k": 20,
    "lr": 0.01,
    "reg": 0.01,

    # ✅ REMOVE single alpha
    "alpha": 0.01,

    # ✅ ADD alpha list
    #"alpha_list": [0.02, 0.05, 0.1, 0.2, 0.3],


    # Training
    "epochs": 10,
    "samples": 50000,

    # Evaluation
    "top_k": 10
}