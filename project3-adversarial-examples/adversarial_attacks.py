from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def softmax(x):
    res = np.exp(x - np.max(x))
    return res / np.sum(res)


def cross_entropy(y, p):
    p = np.clip(p, 1e-12, 1.0)
    log_likelihood = -np.log(p)
    return log_likelihood[y]


def relu(x):
    return np.maximum(x, 0)


class MultiLayerPerceptron:
    def __init__(self):
        self.eps = 0.1

    def load_params(self, params):
        self.W1 = params["fc1.weight"]
        self.b1 = params["fc1.bias"]
        self.W2 = params["fc2.weight"]
        self.b2 = params["fc2.bias"]
        self.W3 = params["fc3.weight"]
        self.b3 = params["fc3.bias"]
        self.W4 = params["fc4.weight"]
        self.b4 = params["fc4.bias"]

    def set_attack_amount(self, eps):
        self.eps = eps

    def forward(self, x):
        W1, W2, W3, W4 = self.W1, self.W2, self.W3, self.W4
        b1, b2, b3, b4 = self.b1, self.b2, self.b3, self.b4

        self.z1 = np.matmul(x, W1) + b1
        self.h1 = relu(self.z1)
        self.z2 = np.matmul(self.h1, W2) + b2
        self.h2 = relu(self.z2)
        self.z3 = np.matmul(self.h2, W3) + b3
        self.h3 = relu(self.z3)
        self.z4 = np.matmul(self.h3, W4) + b4
        self.p = softmax(self.z4)

        return self.p

    def predict(self, x):
        res = self.forward(x)
        return np.argmax(res)

    def gradient(self, x, y):
        p = self.forward(x)
        y_one_hot = np.zeros_like(p)
        y_one_hot[y] = 1

        dL_dz4 = p - y_one_hot
        dL_dh3 = np.matmul(dL_dz4, self.W4.T)
        dL_dz3 = dL_dh3 * (self.h3 > 0)
        dL_dh2 = np.matmul(dL_dz3, self.W3.T)
        dL_dz2 = dL_dh2 * (self.h2 > 0)
        dL_dh1 = np.matmul(dL_dz2, self.W2.T)
        dL_dz1 = dL_dh1 * (self.h1 > 0)
        dL_dx = np.matmul(dL_dz1, self.W1.T)

        return dL_dx

    def attack(self, x, y):
        x_adv = x + self.eps * np.sign(self.gradient(x, y))
        return np.clip(x_adv, 0, 1)

    def mifgsm_attack(self, input, y, epsilon):
        iter = 10
        decay_factor = 1.0
        pert_out = np.copy(input)
        alpha = epsilon / iter
        g = np.zeros_like(input)

        for i in range(iter):
            data_grad = self.gradient(pert_out, y)
            grad_norm = np.linalg.norm(data_grad, ord=1)
            if grad_norm != 0:
                data_grad = data_grad / grad_norm

            g = decay_factor * g + data_grad
            pert_out = pert_out + alpha * np.sign(g)
            pert_out = np.clip(pert_out, input - epsilon, input + epsilon)
            pert_out = np.clip(pert_out, 0, 1)

        return pert_out


def load_data_and_model(data_dir="./data"):
    X_test = np.load(f"{data_dir}/X_test.npy")
    Y_test = np.load(f"{data_dir}/Y_test.npy")

    params = {}
    param_names = [
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
        "fc3.weight",
        "fc3.bias",
        "fc4.weight",
        "fc4.bias",
    ]

    for name in param_names:
        params[name] = np.load(f"{data_dir}/{name}.npy")

    clf = MultiLayerPerceptron()
    clf.load_params(params)
    return X_test, Y_test, clf


def visualize_first_image(X_test, Y_test):
    x, y = X_test[0], Y_test[0]
    print("This is an image of Number", y)
    pixels = x.reshape((28, 28))
    plt.imshow(pixels, cmap="gray")
    plt.show()


def evaluate_clean_accuracy(clf, X_test, Y_test, n_test=1000):
    Y_pred = np.zeros(n_test)
    for i in tqdm(range(n_test), desc="Clean accuracy", leave=False):
        x, _ = X_test[i], Y_test[i]
        Y_pred[i] = clf.predict(x)
    acc = np.sum(Y_pred == Y_test[:n_test]) * 1.0 / n_test
    print("Test accuracy is", acc)
    return acc


def evaluate_fgsm_accuracy(clf, X_test, Y_test, eps=0.1):
    clf.set_attack_amount(eps)

    n_test = len(Y_test)
    X_adv = np.zeros_like(X_test)
    Y_adv_pred = np.zeros(n_test, dtype=int)

    for i in tqdm(range(n_test), desc=f"FGSM eps={eps}", leave=False):
        x, y = X_test[i], int(Y_test[i])
        X_adv[i] = clf.attack(x, y)
        Y_adv_pred[i] = clf.predict(X_adv[i])

    adv_acc = np.mean(Y_adv_pred == Y_test)
    print("FGSM adversarial accuracy at epsilon =", eps, "is", adv_acc)
    return adv_acc


def evaluate_mifgsm_accuracy(clf, X_test, Y_test, eps=0.1):
    n_test = len(Y_test)
    X_adv = np.zeros_like(X_test)
    Y_adv_pred = np.zeros(n_test, dtype=int)

    for i in tqdm(range(n_test), desc=f"MI-FGSM eps={eps}", leave=False):
        x, y = X_test[i], int(Y_test[i])
        X_adv[i] = clf.mifgsm_attack(x, y, eps)
        Y_adv_pred[i] = clf.predict(X_adv[i])

    adv_acc = np.mean(Y_adv_pred == Y_test)
    print("MI-FGSM adversarial accuracy at epsilon =", eps, "is", adv_acc)
    return adv_acc


def compare_fgsm_and_mifgsm(clf, X_test, Y_test, eps_list=(0.05, 0.1, 0.15, 0.2)):
    results = []

    for eps in eps_list:
        fgsm_acc = evaluate_fgsm_accuracy(clf, X_test, Y_test, eps=eps)
        mifgsm_acc = evaluate_mifgsm_accuracy(clf, X_test, Y_test, eps=eps)
        results.append((eps, fgsm_acc, mifgsm_acc))

    return results


def main():
    X_test, Y_test, clf = load_data_and_model()
    evaluate_clean_accuracy(clf, X_test, Y_test, n_test=1000)
    evaluate_fgsm_accuracy(clf, X_test, Y_test, eps=0.1)
    compare_fgsm_and_mifgsm(clf, X_test, Y_test)


if __name__ == "__main__":
    main()
