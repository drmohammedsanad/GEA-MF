import numpy as np


class FairRankingMF:

    def __init__(self, n_users, n_items, k=20, lr=0.01, reg=0.01, alpha=0.1):

        self.n_users = n_users
        self.n_items = n_items
        self.k = k

        self.U = np.random.normal(0, 0.1, (n_users, k))
        self.V = np.random.normal(0, 0.1, (n_items, k))

        self.lr = lr
        self.reg = reg
        self.alpha = alpha

        self.item_groups = None
        self.group_exposure = np.zeros(2)

    def predict(self, u, i):
        return np.dot(self.U[u], self.V[i])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update_exposure(self, R, K=10):

        exposure = np.zeros(2)

        for u in range(R.shape[0]):
            scores = self.U[u].dot(self.V.T)
            top_k = np.argsort(scores)[-K:]

            for i in top_k:
                g = int(self.item_groups[i])
                exposure[g] += 1

        self.group_exposure = exposure

    def train(self, samples, R, n_epochs=10, top_k=10):

        for epoch in range(n_epochs):

            np.random.shuffle(samples)

            # ✅ STEP 1: PURE BPR TRAINING
            for (u, i, j) in samples:

                x_ui = self.predict(u, i)
                x_uj = self.predict(u, j)

                x = x_ui - x_uj
                s = self.sigmoid(x)
                grad = (1 - s)

                u_old = self.U[u].copy()

                self.U[u] += self.lr * (
                    grad * (self.V[i] - self.V[j])
                    - self.reg * self.U[u]
                )

                self.V[i] += self.lr * (
                    grad * u_old - self.reg * self.V[i]
                )

                self.V[j] += self.lr * (
                    -grad * u_old - self.reg * self.V[j]
                )

            # ✅ STEP 2: COMPUTE EXPOSURE
            self.update_exposure(R, top_k)

            # ✅ STEP 3: GLOBAL FAIRNESS CORRECTION
            exp0, exp1 = self.group_exposure

            imbalance = (exp0 - exp1) / (exp0 + exp1 + 1e-6)

            for i in range(self.n_items):
                g = int(self.item_groups[i])

                if g == 0:
                    self.V[i] -= self.lr * self.alpha * 9 * imbalance
                else:
                    self.V[i] += self.lr * self.alpha * 9 * imbalance

            print(f"[FR-MF GLOBAL] Epoch {epoch+1}/{n_epochs}")
            print(f"   Exposure: {self.group_exposure}")
            print(f"   Imbalance: {imbalance:.6f}")
