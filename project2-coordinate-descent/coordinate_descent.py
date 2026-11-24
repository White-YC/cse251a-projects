import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def load_wine_data(data_path):
    data = np.loadtxt(data_path, delimiter=",")
    mask = (data[:, 0] == 1) | (data[:, 0] == 2)
    X = data[mask, 1:]
    y = (data[mask, 0] == 2).astype(np.float64)
    return X, y


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss(w, X, y):
    z = np.clip(X @ w, -500, 500)
    return np.sum(np.log(1 + np.exp(z)) - y * z)


def gradient_and_hessian_diag(w, X, y):
    p = sigmoid(X @ w)
    g = X.T @ (p - y)
    h_diag = np.sum(X**2 * p[:, np.newaxis] * (1 - p)[:, np.newaxis], axis=0)
    return g, h_diag


def coordinate_update(w, j, g, h_diag):
    h_j = h_diag[j]
    if h_j < 1e-10:
        return w
    w_new = w.copy()
    w_new[j] = w[j] - g[j] / h_j
    return w_new


def gauss_southwell_coordinate_descent(X, y, n_iter):
    n, d = X.shape
    w = np.zeros(d)
    losses = [logistic_loss(w, X, y)]
    for _ in range(n_iter - 1):
        g, h_diag = gradient_and_hessian_diag(w, X, y)
        j = np.argmax(np.abs(g))
        w = coordinate_update(w, j, g, h_diag)
        losses.append(logistic_loss(w, X, y))
    return w, np.array(losses)


def random_coordinate_descent(X, y, n_iter, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n, d = X.shape
    w = np.zeros(d)
    losses = [logistic_loss(w, X, y)]
    for _ in range(n_iter - 1):
        g, h_diag = gradient_and_hessian_diag(w, X, y)
        j = rng.integers(0, d)
        w = coordinate_update(w, j, g, h_diag)
        losses.append(logistic_loss(w, X, y))
    return w, np.array(losses)


def hard_threshold(w, k):
    if k >= len(w):
        return w
    idx = np.argsort(np.abs(w))[::-1]
    w_sparse = np.zeros_like(w)
    w_sparse[idx[:k]] = w[idx[:k]]
    return w_sparse


def sparse_coordinate_descent(X, y, k, n_iter):
    n, d = X.shape
    if k > d:
        k = d
    w = np.zeros(d)
    losses = [logistic_loss(w, X, y)]
    for _ in range(n_iter - 1):
        g, h_diag = gradient_and_hessian_diag(w, X, y)
        j = np.argmax(np.abs(g))
        w = coordinate_update(w, j, g, h_diag)
        w = hard_threshold(w, k)
        losses.append(logistic_loss(w, X, y))
    return w, np.array(losses)


def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "wine_data", "wine.data")
    X_raw, y = load_wine_data(data_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = add_bias(X_scaled)
    n, d = X.shape
    print(f"Data: {n} samples, {d} features (including bias)")

    lr_full = LogisticRegression(
        penalty=None,
        max_iter=50000,
        tol=1e-12,
        solver="lbfgs",
        fit_intercept=True,
        random_state=42,
    )
    lr_full.fit(X_scaled, y)
    w_sklearn = np.concatenate([[lr_full.intercept_[0]], lr_full.coef_[0]])
    L_star = logistic_loss(w_sklearn, X, y)
    print(f"Reference L* (sklearn): {L_star:.6f}")

    n_iter = 1000
    rng = np.random.default_rng(42)
    w_gs, losses_gs = gauss_southwell_coordinate_descent(X, y, n_iter)
    rng = np.random.default_rng(42)
    w_rand, losses_rand = random_coordinate_descent(X, y, n_iter, rng)
    print(f"Gauss-Southwell final loss: {losses_gs[-1]:.6f}")
    print(f"Random final loss: {losses_rand[-1]:.6f}")
    print(f"Gap to L*: GS={losses_gs[-1] - L_star:.6e}, Random={losses_rand[-1] - L_star:.6e}")

    plt.figure(figsize=(8, 5))
    plt.plot(losses_gs, label="Gauss-Southwell", color="C0")
    plt.plot(losses_rand, label="Random", color="C1", alpha=0.8)
    plt.axhline(y=L_star, color="gray", linestyle="--", label=r"$L^*$")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Coordinate Descent: Loss vs Iteration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "loss_comparison.pdf"))
    plt.close()

    k_values = [1, 2, 3, 5, 7, 10, 13, 14]
    results = []
    for k in k_values:
        w_sparse, losses_sparse = sparse_coordinate_descent(X, y, k, n_iter)
        results.append((k, losses_sparse[-1], np.sum(w_sparse != 0)))
    print("\nSparse coordinate descent: final loss vs k")
    print(f"{'k':>4}  {'Final Loss':>12}  {'nnz (<=k)':>10}")
    for k, loss, nz in results:
        print(f"{k:>4}  {loss:>12.6f}  {nz:>10}")

    with open(os.path.join(script_dir, "sparse_results.txt"), "w") as f:
        f.write(f"{'k':>4}  {'Final Loss':>12}  {'nnz (<=k)':>10}\n")
        for k, loss, nz in results:
            f.write(f"{k:>4}  {loss:>12.6f}  {nz:>10}\n")


if __name__ == "__main__":
    main()