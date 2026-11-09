"""
Bernoulli Mixture Model optimized via the EM Algorithm on binarized MNIST images.
"""
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

NUM_SAMPLES = 1800
LABELS = (0, 5, 7)


def load_data(
    num_samples: int = NUM_SAMPLES,
    labels: tuple[int, ...] = LABELS,
    data_dir: str = "./data",
) -> np.ndarray:
    """Load MNIST, filter target digits, flatten to 784-d, and binarize at 0.5."""
    mnist_train = datasets.MNIST(root=data_dir, train=True, download=True)
    x_all = mnist_train.data.numpy()     # (60000, 28, 28), uint8 [0, 255]
    t_all = mnist_train.targets.numpy()  # (60000,)

    mask = np.isin(t_all, labels)
    x_filtered = x_all[mask][:num_samples]

    # Normalize to [0, 1], flatten to (N, 784), and binarize
    x_norm = x_filtered.astype(np.float64) / 255.0
    x_flat = x_norm.reshape(x_norm.shape[0], -1)
    binary_data = (x_flat > 0.5).astype(np.float64)
    return binary_data


def np_log(x: np.ndarray) -> np.ndarray:
    """Numerically stable natural log with clipping to avoid log(0) = -inf."""
    return np.log(np.clip(x, a_min=1e-323, a_max=1e10))


def log_bernoulli_matrix(data: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Compute log p(x_i | mu_k) for all samples i and components k simultaneously.
    data: (N, D), mu: (K, D) -> returns: (N, K)
    """
    return data @ np_log(mu.T) + (1.0 - data) @ np_log(1.0 - mu.T)


def log_likelihood(data: np.ndarray, mu: np.ndarray, pi: np.ndarray) -> float:
    """
    Compute total log-likelihood across all N samples using the log-sum-exp trick.
    data: (N, D), mu: (K, D), pi: (K,)
    """
    log_prob = log_bernoulli_matrix(data, mu) + np_log(pi)[np.newaxis, :]  # (N, K)
    max_log_prob = np.max(log_prob, axis=1, keepdims=True)                 # (N, 1)
    log_likely_per_sample = max_log_prob.squeeze(1) + np_log(
        np.sum(np.exp(log_prob - max_log_prob), axis=1)
    )
    return float(np.sum(log_likely_per_sample))


def responsibility(data: np.ndarray, mu: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """
    E-step: Compute posterior responsibilities gamma(z_{ik}).
    data: (N, D), mu: (K, D), pi: (K,) -> returns: (N, K)
    """
    log_p = log_bernoulli_matrix(data, mu)                 # (N, K)
    log_p -= np.max(log_p, axis=1, keepdims=True)          # Stabilize exp
    weighted_p = pi[np.newaxis, :] * np.exp(log_p)         # (N, K)
    gamma_mat = weighted_p / np.sum(weighted_p, axis=1, keepdims=True)
    return gamma_mat


def visualize(mu: np.ndarray, iteration: int | None = None) -> None:
    """Plot the learned Bernoulli mean templates as 28x28 grayscale images."""
    n_components = mu.shape[0]
    fig = plt.figure(figsize=(3 * n_components, 3))
    if iteration is not None:
        fig.suptitle(f"Iteration {iteration}")
    for k in range(n_components):
        ax = fig.add_subplot(1, n_components, k + 1, xticks=[], yticks=[])
        ax.imshow(mu[k].reshape(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Component {k + 1}")
    plt.tight_layout()
    plt.show()


def EM_algorithm_bernoulli(
    data: np.ndarray,
    pi_init: list[float] | None = None,
    threshold: float = 0.001,
    max_iter: int = 50,
    plot_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run EM for a Bernoulli Mixture Model and return (mu, pi)."""
    n_samples, n_features = data.shape
    pi = np.array(pi_init if pi_init is not None else [0.3, 0.3, 0.4], dtype=np.float64)
    n_components = len(pi)

    # Initialize mu in (0.25, 0.75) to avoid extreme initial log-probabilities
    mu = np.random.uniform(0.25, 0.75, size=(n_components, n_features))

    n_iter = 0
    likely = log_likelihood(data, mu, pi) / n_samples
    print(f"Iteration: {n_iter}, log_likelihood: {likely:.4f}")
    if plot_progress:
        visualize(mu, iteration=n_iter)

    while True:
        n_iter += 1

        # E-step: Posterior responsibilities (N, K)
        gamma_mat = responsibility(data, mu, pi)
        n_k = np.sum(gamma_mat, axis=0)  # (K,)

        # M-step: Update mixing weights pi (K,) and Bernoulli means mu (K, D)
        pi = n_k / n_samples
        mu = (gamma_mat.T @ data) / n_k[:, np.newaxis]

        likely_before = likely
        likely = log_likelihood(data, mu, pi) / n_samples
        delta = likely - likely_before

        print(f"Iteration: {n_iter}, log_likelihood: {likely:.4f}, pi: {np.round(pi, 4)}")
        if plot_progress:
            visualize(mu, iteration=n_iter)

        if abs(delta) < threshold or n_iter >= max_iter:
            break

    return mu, pi


if __name__ == "__main__":
    binary_data = load_data()
    mu_opt, pi_opt = EM_algorithm_bernoulli(binary_data)
