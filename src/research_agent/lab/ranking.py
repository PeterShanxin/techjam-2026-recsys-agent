"""Within-user grouping and gradient-agnostic FM machinery. Instruments, not a ranker.

Rationale (research-space audit, 2026-08-31): every attempt at a ranking-aligned
objective so far failed on runtime, not on science. The pairwise candidates built
O(pairs) Python/SGD loops over ~1.9M sampled pairs per epoch and hit the experiment
timeout before producing a metric. That is an implementation failure, so it never
counted as evidence about ranking objectives.

This module removes that runtime blind spot and nothing else:

- ``user_groups`` gives the vectorized within-user segmentation of an encoded split,
  so a group-structured objective costs O(rows) instead of O(pairs).
- ``GradientFM`` exposes the official FM forward pass plus its Adam parameter update
  driven by an arbitrary per-row upstream gradient supplied by the caller.

``GradientFM`` deliberately contains **no loss function**. It cannot train anything on
its own. The objective, the targets, the sampling policy, the regularization strength
and the stopping rule are all research decisions and remain the caller's to make.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


@dataclass(frozen=True)
class UserGroups:
    """Contiguous within-user segmentation of one encoded split.

    ``order`` reindexes the split into user-contiguous order. ``group_start`` and
    ``group_sizes`` describe the segments in that reordered space, which is what
    ``numpy.add.reduceat`` / ``numpy.maximum.reduceat`` consume directly.
    """

    order: np.ndarray
    group_start: np.ndarray
    group_sizes: np.ndarray
    group_ids: np.ndarray

    @property
    def n_groups(self) -> int:
        return int(len(self.group_start))

    @property
    def n_rows(self) -> int:
        return int(len(self.order))

    def broadcast(self, per_group: np.ndarray) -> np.ndarray:
        """Expand one value per group back to one value per row."""
        values = np.asarray(per_group)
        if len(values) != self.n_groups:
            raise ValueError(f"expected {self.n_groups} group values, got {len(values)}")
        return np.repeat(values, self.group_sizes)

    def group_sum(self, per_row: np.ndarray) -> np.ndarray:
        """Sum a per-row vector inside each group."""
        values = np.asarray(per_row)
        if len(values) != self.n_rows:
            raise ValueError(f"expected {self.n_rows} row values, got {len(values)}")
        if self.n_groups == 0:
            return np.zeros(0, dtype=values.dtype)
        return np.add.reduceat(values, self.group_start)

    def group_max(self, per_row: np.ndarray) -> np.ndarray:
        """Maximum of a per-row vector inside each group."""
        values = np.asarray(per_row)
        if len(values) != self.n_rows:
            raise ValueError(f"expected {self.n_rows} row values, got {len(values)}")
        if self.n_groups == 0:
            return np.zeros(0, dtype=values.dtype)
        return np.maximum.reduceat(values, self.group_start)

    def select(self, group_index: Sequence[int] | np.ndarray) -> np.ndarray:
        """Row positions (in reordered space) for a subset of groups.

        Use this to build user-respecting minibatches: sample group indices, then
        gather their rows. Groups are never split across batches.
        """
        picked = np.asarray(group_index, dtype=np.int64)
        if picked.size == 0:
            return np.zeros(0, dtype=np.int64)
        sizes = self.group_sizes[picked]
        starts = self.group_start[picked]
        offsets = np.repeat(starts, sizes)
        within = np.arange(int(sizes.sum()), dtype=np.int64) - np.repeat(
            np.concatenate(([0], np.cumsum(sizes)[:-1])), sizes
        )
        return offsets + within


def user_groups(users: Iterable[Any]) -> UserGroups:
    """Segment an encoded split by user id, preserving official row order inside a user.

    ``users`` is the third element of a ``data.encode()`` split tuple. Sorting is
    stable, so rows keep their official relative order within each user.
    """
    keys = np.asarray(list(users))
    if keys.size == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        return UserGroups(empty_i, empty_i, empty_i, np.zeros(0, dtype=keys.dtype))
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    boundary = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1])))
    ends = np.concatenate((boundary[1:], [len(ordered)]))
    return UserGroups(
        order=order.astype(np.int64),
        group_start=boundary.astype(np.int64),
        group_sizes=(ends - boundary).astype(np.int64),
        group_ids=ordered[boundary],
    )


class GradientFM:
    """Official FM forward pass plus Adam, driven by a caller-supplied row gradient.

    This is plumbing. It has no loss, no targets and no training loop. ``apply`` takes
    ``dL/dlogit`` for each row and performs one Adam step on the FM parameters. Pass a
    pointwise BCE gradient and it reproduces the organizer FM; pass anything else and
    it trains that objective instead. Which objective is worth training is exactly the
    research question and this class does not answer it.

    Parameters mirror ``baseline.FM`` so a candidate can move weights between the two.
    """

    def __init__(
        self,
        dim: int,
        *,
        k: int = 16,
        lr: float = 0.001,
        l2: float = 1e-6,
        seed: int = 0,
        init_scale: float = 0.01,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        rng = np.random.default_rng(seed)
        self.dim = int(dim)
        self.k = int(k)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.V = rng.normal(0.0, init_scale, (self.dim, self.k)).astype(np.float32)
        self.W = np.zeros(self.dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (logits, per-field embeddings, embedding sum). Same algebra as baseline.FM."""
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S**2).sum(1) - (E**2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        return np.concatenate(
            [self.logits(X[i : i + bs])[0] for i in range(0, len(X), bs)]
        )

    def apply(
        self,
        X: np.ndarray,
        grad_logits: np.ndarray,
        *,
        update_bias: bool = True,
    ) -> None:
        """One Adam step given ``dL/dlogit`` per row.

        ``grad_logits`` must already carry whatever normalization the objective wants
        (per-row, per-group, per-batch). This method does not rescale it.
        """
        rows = np.asarray(grad_logits, dtype=np.float32)
        if rows.shape != (len(X),):
            raise ValueError(f"grad_logits must have shape ({len(X)},), got {rows.shape}")
        if not np.all(np.isfinite(rows)):
            raise ValueError("grad_logits contains non-finite values")
        _, E, S = self.logits(X)
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, rows[:, None])
        np.add.at(gV, X, rows[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= ADAM_BETA1
            M += (1 - ADAM_BETA1) * G
            Vv *= ADAM_BETA2
            Vv += (1 - ADAM_BETA2) * (G * G)
            P -= self.lr * (M / (1 - ADAM_BETA1**self.t)) / (
                np.sqrt(Vv / (1 - ADAM_BETA2**self.t)) + ADAM_EPS
            )
        if update_bias:
            self.b -= np.float32(self.lr * rows.sum())

    def state(self) -> tuple[np.ndarray, np.ndarray, np.float32]:
        return (self.V.copy(), self.W.copy(), np.float32(self.b))

    def load_state(self, state: tuple[np.ndarray, np.ndarray, Any]) -> None:
        V, W, b = state
        self.V = np.asarray(V, dtype=np.float32).copy()
        self.W = np.asarray(W, dtype=np.float32).copy()
        self.b = np.float32(b)
