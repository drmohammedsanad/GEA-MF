import numpy as np


class BPR:
    """
    Standard Bayesian Personalized Ranking (BPR)

    Used as ranking baseline (NO fairness)
    """

    def __init__(self, n_users, n_items, k=20, lr=0.01, reg=0.01):
        self.U = np.random.normal(0, 0.1, (n_users, k))
        self.V = np.random.normal(0, 0.1, (n_items, k))

        self.lr = lr
        self.reg = reg

    def predict(self, u, i):
        return np.dot(self.U[u], self.V[i])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, samples, epochs=10):
        for epoch in range(epochs):
            np.random.shuffle(samples)

            for u, i, j in samples:
                x = self.predict(u, i) - self.predict(u, j)
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

            print(f"[BPR] Epoch {epoch+1}")