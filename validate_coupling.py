"""Check the sparse implementation against the direct greedy construction."""

import argparse
from pathlib import Path
import time

import numpy as np

from imec import Belief, GreedyCoupling, uniform_belief


def dense_greedy(prior, channel):
    p = prior.copy()
    q = channel.copy()
    matrix = np.zeros((len(p), len(q)))
    while p.max() > 0 and q.max() > 0:
        row, col = np.argmax(p), np.argmax(q)
        mass = min(p[row], q[col])
        matrix[row, col] = mass
        p[row] -= mass
        q[col] -= mass
    return matrix


def entropy(probabilities):
    positive = probabilities[probabilities > 0]
    return -float(np.dot(positive, np.log2(positive)))


def verify_column(coupling, belief, channel, joint, column):
    selected, posterior = coupling.update(belief, channel, observed=column)
    expected = joint[:, column] / joint[:, column].sum()
    observed = np.zeros_like(expected)
    observed[posterior.states] = posterior.probabilities
    np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(posterior.entropy, entropy(expected), atol=1e-14)
    assert selected == column and posterior.mode == np.argmax(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("coupling.so"))
    args = parser.parse_args()
    coupling = GreedyCoupling(args.library)
    rng = np.random.default_rng(24003)
    checks = 0
    for _ in range(100):
        # Dyadic probabilities make ties and zero residuals exactly representable.
        p = rng.multinomial(256, np.full(32, 1 / 32)) / 256
        q = rng.multinomial(256, np.full(7, 1 / 7)) / 256
        indices = rng.permutation(np.flatnonzero(p)).astype(np.int32)
        belief = Belief(indices, p[indices], entropy(p), int(np.argmax(p)))
        joint = dense_greedy(p, q)
        np.testing.assert_array_equal(joint.sum(axis=0), q)
        np.testing.assert_array_equal(joint.sum(axis=1), p)
        for column in range(len(q)):
            verify_column(coupling, belief, q, joint, column)
            checks += 1
        state = int(rng.choice(indices))
        column, encoded = coupling.update(belief, q, state=state, coin=rng.random())
        _, decoded = coupling.update(belief, q, observed=column)
        assert joint[state, column] > 0
        np.testing.assert_array_equal(encoded.states, decoded.states)
        np.testing.assert_array_equal(encoded.probabilities, decoded.probabilities)

    prior = uniform_belief(20)
    channel = np.arange(40, 0, -1, dtype=np.float64)
    channel /= channel.sum()
    start = time.perf_counter()
    column, encoded = coupling.update(prior, channel, state=736711, coin=0.42)
    _, decoded = coupling.update(prior, channel, observed=column)
    assert 736711 in decoded.states
    np.testing.assert_array_equal(encoded.states, decoded.states)
    np.testing.assert_array_equal(encoded.probabilities, decoded.probabilities)
    np.testing.assert_allclose(decoded.probabilities.sum(), 1.0, atol=1e-13)
    print(f"{checks} dense-reference posterior checks and 100 encoder/decoder checks passed")
    print(f"20-bit sparse posterior check passed in {time.perf_counter() - start:.3f} s")


if __name__ == "__main__":
    main()
