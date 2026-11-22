"""Sparse iterative minimum entropy coupling for fixed-size messages."""

import ctypes as ct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TOP_K = 40
MESSAGE_BITS = 80
BLOCK_BITS = 20
STOP_ENTROPY = 0.1


@dataclass
class Belief:
    states: np.ndarray
    probabilities: np.ndarray
    entropy: float
    mode: int


def uniform_belief(block_bits):
    size = 1 << block_bits
    return Belief(
        np.arange(size, dtype=np.int32), np.full(size, 1.0 / size), float(block_bits), 0
    )


class GreedyCoupling:
    def __init__(self, library):
        self.library = ct.CDLL(str(Path(library).resolve()))
        f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        i32 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
        self.call = self.library.d203_couple
        self.call.restype = ct.c_int
        self.call.argtypes = [
            ct.c_int, i32, f64, ct.c_int, f64, ct.c_int, ct.c_double, ct.c_int,
            i32, i32, f64, ct.POINTER(ct.c_double),
        ]

    def update(self, belief, probabilities, *, state=-1, coin=0.0, observed=-1):
        """Sample a token column or update from an already received column."""
        size = len(belief.states)
        states = np.empty(size, dtype=np.int32)
        weights = np.empty(size, dtype=np.float64)
        selected = np.empty(2, dtype=np.int32)
        entropy = ct.c_double()
        count = self.call(
            size, belief.states, belief.probabilities, len(probabilities), probabilities,
            state, coin, observed, selected, states, weights, ct.byref(entropy),
        )
        if count <= 0:
            raise ValueError(f"Invalid coupling posterior (status {count})")
        updated = Belief(states[:count].copy(), weights[:count].copy(), entropy.value, int(selected[1]))
        return int(selected[0]), updated


class Trajectory:
    def __init__(self, initial, *, block_bits, message=None, random=None):
        self.block_bits = block_bits
        self.beliefs = [initial] * (MESSAGE_BITS // block_bits)
        self.message = message
        self.random = random
        self.tokens = []
        self.entropies = []
        self.status = "running"
        self.failure = None
        if message is None:
            self.blocks = None
        else:
            self.blocks = [
                int(message[i : i + block_bits], 2) for i in range(0, MESSAGE_BITS, block_bits)
            ]

    @property
    def max_entropy(self):
        return max(belief.entropy for belief in self.beliefs)

    def advance(self, coupling, labels, probabilities, observed_token=None):
        index = max(range(len(self.beliefs)), key=lambda i: self.beliefs[i].entropy)
        prior = self.beliefs[index]
        if observed_token is None:
            column, updated = coupling.update(
                prior, probabilities, state=self.blocks[index], coin=self.random.random()
            )
        else:
            matches = np.flatnonzero(labels == observed_token)
            if len(matches) != 1:
                raise ValueError("Received token is outside decoder top-40")
            column, updated = coupling.update(prior, probabilities, observed=int(matches[0]))

        self.beliefs[index] = updated
        self.tokens.append(int(labels[column]))
        self.entropies.append(float(-np.dot(probabilities, np.log2(probabilities))))
        if self.max_entropy < STOP_ENTROPY:
            self.status = "complete"

    def recovered_message(self):
        return "".join(f"{belief.mode:0{self.block_bits}b}" for belief in self.beliefs)
