from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


SUPPORTED_SEARCH_BACKENDS = [
    "exact",
    "hnsw",
    "faiss_ivf_flat",
    "faiss_ivf_pq",
]

SUPPORTED_SEARCH_METRICS = [
    "cosine",
    "inner_product",
]


@dataclass
class SearchConfig:
    backend: str = "exact"
    metric: str = "cosine"
    candidate_k: int = 50
    rerank: bool = True
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    ivf_nlist: int = 100
    ivf_nprobe: int = 10
    pq_m: int = 16
    pq_bits: int = 8
    train_limit: int = 10000

    @classmethod
    def from_namespace(cls, args) -> "SearchConfig":
        return cls(
            backend=args.search_backend,
            metric=args.search_metric,
            candidate_k=args.search_candidate_k,
            rerank=bool(args.search_rerank),
            hnsw_m=args.search_hnsw_m,
            hnsw_ef_construction=args.search_hnsw_ef_construction,
            hnsw_ef_search=args.search_hnsw_ef_search,
            ivf_nlist=args.search_ivf_nlist,
            ivf_nprobe=args.search_ivf_nprobe,
            pq_m=args.search_pq_m,
            pq_bits=args.search_pq_bits,
            train_limit=args.search_train_limit,
        )

    def validate(self) -> None:
        if self.backend not in SUPPORTED_SEARCH_BACKENDS:
            raise ValueError(f"Unsupported search backend: {self.backend}")
        if self.metric not in SUPPORTED_SEARCH_METRICS:
            raise ValueError(f"Unsupported search metric: {self.metric}")
        if self.candidate_k < 0:
            raise ValueError("search candidate_k must be >= 0")
        if self.hnsw_m <= 0:
            raise ValueError("search hnsw_m must be > 0")
        if self.hnsw_ef_construction <= 0:
            raise ValueError("search hnsw_ef_construction must be > 0")
        if self.hnsw_ef_search <= 0:
            raise ValueError("search hnsw_ef_search must be > 0")
        if self.ivf_nlist <= 0:
            raise ValueError("search ivf_nlist must be > 0")
        if self.ivf_nprobe <= 0:
            raise ValueError("search ivf_nprobe must be > 0")
        if self.pq_m <= 0:
            raise ValueError("search pq_m must be > 0")
        if self.pq_bits <= 0:
            raise ValueError("search pq_bits must be > 0")
        if self.train_limit < 0:
            raise ValueError("search train_limit must be >= 0")

    def effective_candidate_k(self, top_k: int, num_references: int) -> int:
        base_k = top_k
        if self.rerank:
            base_k = max(top_k, self.candidate_k or top_k)
        return max(1, min(base_k, num_references))


@dataclass
class SearchResult:
    indices: list[int]
    scores: list[float]
    all_scores: np.ndarray | None = None


class SearchBackend(Protocol):
    def search(self, query_descriptor: np.ndarray, top_k: int) -> SearchResult:
        ...

    def describe(self) -> str:
        ...


def _normalize_metric(metric: str) -> str:
    if metric == "inner_product":
        return "inner_product"
    return "cosine"


def _prepare_descriptors(descriptors: np.ndarray) -> np.ndarray:
    prepared = np.asarray(descriptors, dtype=np.float32)
    if prepared.ndim != 2:
        raise ValueError(f"Expected descriptors with shape [N, D], got {prepared.shape}")
    if len(prepared) == 0:
        raise ValueError("Search backend cannot be built for an empty descriptor matrix")
    return np.ascontiguousarray(prepared)


def _prepare_query(query_descriptor: np.ndarray) -> np.ndarray:
    query = np.asarray(query_descriptor, dtype=np.float32).reshape(-1)
    return np.ascontiguousarray(query)


def _select_top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    if top_k >= len(scores):
        return np.argsort(scores)[::-1]
    candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
    candidate_scores = scores[candidate_indices]
    return candidate_indices[np.argsort(candidate_scores)[::-1]]


def _exact_scores_for_indices(descriptors: np.ndarray, query: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = descriptors[indices]
    return (selected @ query.reshape(-1, 1)).reshape(-1)


def _rerank_exact(
    descriptors: np.ndarray,
    query: np.ndarray,
    candidate_indices: np.ndarray,
    top_k: int,
) -> tuple[list[int], list[float]]:
    if len(candidate_indices) == 0:
        return [], []
    exact_scores = _exact_scores_for_indices(descriptors, query, candidate_indices)
    order = _select_top_indices(exact_scores, min(top_k, len(candidate_indices)))
    reranked_indices = candidate_indices[order]
    reranked_scores = exact_scores[order]
    return reranked_indices.astype(int).tolist(), reranked_scores.astype(float).tolist()


class ExactSearchBackend:
    def __init__(self, descriptors: np.ndarray, config: SearchConfig):
        self.descriptors = _prepare_descriptors(descriptors)
        self.config = config

    def search(self, query_descriptor: np.ndarray, top_k: int) -> SearchResult:
        query = _prepare_query(query_descriptor)
        scores = (self.descriptors @ query.reshape(-1, 1)).reshape(-1)
        order = _select_top_indices(scores, min(top_k, len(scores)))
        return SearchResult(
            indices=order.astype(int).tolist(),
            scores=scores[order].astype(float).tolist(),
            all_scores=scores,
        )

    def describe(self) -> str:
        return "exact (full scan + partial top-k)"


class HNSWSearchBackend:
    def __init__(self, descriptors: np.ndarray, config: SearchConfig):
        self.descriptors = _prepare_descriptors(descriptors)
        self.config = config
        self._hnswlib = self._import_hnswlib()
        self.index = self._build_index()

    def _import_hnswlib(self):
        try:
            import hnswlib  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "hnswlib is required for search_backend=hnsw. Install it with `pip install hnswlib`."
            ) from exc
        return hnswlib

    def _build_index(self):
        space = "cosine" if _normalize_metric(self.config.metric) == "cosine" else "ip"
        index = self._hnswlib.Index(space=space, dim=int(self.descriptors.shape[1]))
        index.init_index(
            max_elements=int(self.descriptors.shape[0]),
            ef_construction=int(self.config.hnsw_ef_construction),
            M=int(self.config.hnsw_m),
        )
        index.add_items(self.descriptors, np.arange(self.descriptors.shape[0]))
        index.set_ef(max(int(self.config.hnsw_ef_search), int(self.config.candidate_k or 1)))
        return index

    def search(self, query_descriptor: np.ndarray, top_k: int) -> SearchResult:
        query = _prepare_query(query_descriptor)
        candidate_k = self.config.effective_candidate_k(top_k, len(self.descriptors))
        self.index.set_ef(max(int(self.config.hnsw_ef_search), candidate_k))
        labels, distances = self.index.knn_query(query.reshape(1, -1), k=candidate_k)
        candidate_indices = np.asarray(labels[0], dtype=np.int32)
        candidate_distances = np.asarray(distances[0], dtype=np.float32)
        valid_mask = candidate_indices >= 0
        candidate_indices = candidate_indices[valid_mask]
        candidate_distances = candidate_distances[valid_mask]

        if self.config.rerank:
            indices, scores = _rerank_exact(self.descriptors, query, candidate_indices, top_k)
        else:
            approx_scores = 1.0 - candidate_distances
            keep = min(top_k, len(candidate_indices))
            indices = candidate_indices[:keep].astype(int).tolist()
            scores = approx_scores[:keep].astype(float).tolist()

        return SearchResult(indices=indices, scores=scores, all_scores=None)

    def describe(self) -> str:
        return (
            f"hnsw(metric={self.config.metric}, M={self.config.hnsw_m}, "
            f"ef_construction={self.config.hnsw_ef_construction}, "
            f"ef_search={self.config.hnsw_ef_search}, rerank={self.config.rerank})"
        )


class _FaissBaseBackend:
    def __init__(self, descriptors: np.ndarray, config: SearchConfig):
        self.descriptors = _prepare_descriptors(descriptors)
        self.config = config
        self.faiss = self._import_faiss()
        self.index = self._build_index()

    def _import_faiss(self):
        try:
            import faiss  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "faiss is required for FAISS search backends. Install it with `pip install faiss-cpu`."
            ) from exc
        return faiss

    def _training_descriptors(self) -> np.ndarray:
        train_limit = int(self.config.train_limit)
        if train_limit <= 0 or len(self.descriptors) <= train_limit:
            return self.descriptors
        step = max(1, len(self.descriptors) // train_limit)
        return np.ascontiguousarray(self.descriptors[::step][:train_limit])

    def _effective_nlist(self) -> int:
        return max(1, min(int(self.config.ivf_nlist), len(self.descriptors)))

    def _effective_nprobe(self, nlist: int) -> int:
        return max(1, min(int(self.config.ivf_nprobe), nlist))

    def _metric_type(self):
        return self.faiss.METRIC_INNER_PRODUCT

    def _search_index(self, query: np.ndarray, candidate_k: int) -> tuple[np.ndarray, np.ndarray]:
        distances, indices = self.index.search(query.reshape(1, -1), candidate_k)
        return np.asarray(indices[0], dtype=np.int32), np.asarray(distances[0], dtype=np.float32)

    def search(self, query_descriptor: np.ndarray, top_k: int) -> SearchResult:
        query = _prepare_query(query_descriptor)
        candidate_k = self.config.effective_candidate_k(top_k, len(self.descriptors))
        candidate_indices, candidate_scores = self._search_index(query, candidate_k)
        valid_mask = candidate_indices >= 0
        candidate_indices = candidate_indices[valid_mask]
        candidate_scores = candidate_scores[valid_mask]

        if self.config.rerank:
            indices, scores = _rerank_exact(self.descriptors, query, candidate_indices, top_k)
        else:
            keep = min(top_k, len(candidate_indices))
            indices = candidate_indices[:keep].astype(int).tolist()
            scores = candidate_scores[:keep].astype(float).tolist()

        return SearchResult(indices=indices, scores=scores, all_scores=None)


class FaissIVFFlatSearchBackend(_FaissBaseBackend):
    def _build_index(self):
        dim = int(self.descriptors.shape[1])
        training_descriptors = self._training_descriptors()
        nlist = max(1, min(self._effective_nlist(), len(training_descriptors)))
        quantizer = self.faiss.IndexFlatIP(dim)
        index = self.faiss.IndexIVFFlat(quantizer, dim, nlist, self._metric_type())
        index.train(training_descriptors)
        index.add(self.descriptors)
        index.nprobe = self._effective_nprobe(nlist)
        return index

    def describe(self) -> str:
        return (
            f"faiss_ivf_flat(metric={self.config.metric}, nlist={self.index.nlist}, "
            f"nprobe={self.index.nprobe}, rerank={self.config.rerank})"
        )


def _choose_effective_pq_m(dim: int, preferred_m: int) -> int:
    preferred = max(1, preferred_m)
    if dim % preferred == 0:
        return preferred
    for candidate in range(preferred - 1, 0, -1):
        if dim % candidate == 0:
            return candidate
    return 1


class FaissIVFPQSearchBackend(_FaissBaseBackend):
    def _build_index(self):
        dim = int(self.descriptors.shape[1])
        training_descriptors = self._training_descriptors()
        nlist = max(1, min(self._effective_nlist(), len(training_descriptors)))
        pq_m = _choose_effective_pq_m(dim, int(self.config.pq_m))
        quantizer = self.faiss.IndexFlatIP(dim)
        index = self.faiss.IndexIVFPQ(
            quantizer,
            dim,
            nlist,
            pq_m,
            int(self.config.pq_bits),
            self._metric_type(),
        )
        index.train(training_descriptors)
        index.add(self.descriptors)
        index.nprobe = self._effective_nprobe(nlist)
        self._effective_pq_m = pq_m
        return index

    def describe(self) -> str:
        return (
            f"faiss_ivf_pq(metric={self.config.metric}, nlist={self.index.nlist}, "
            f"nprobe={self.index.nprobe}, pq_m={self._effective_pq_m}, "
            f"pq_bits={self.config.pq_bits}, rerank={self.config.rerank})"
        )


def create_search_backend(descriptors: np.ndarray, config: SearchConfig | None = None) -> SearchBackend:
    search_config = config or SearchConfig()
    search_config.validate()

    if search_config.backend == "exact":
        return ExactSearchBackend(descriptors, search_config)
    if search_config.backend == "hnsw":
        return HNSWSearchBackend(descriptors, search_config)
    if search_config.backend == "faiss_ivf_flat":
        return FaissIVFFlatSearchBackend(descriptors, search_config)
    if search_config.backend == "faiss_ivf_pq":
        return FaissIVFPQSearchBackend(descriptors, search_config)

    raise ValueError(f"Unsupported search backend: {search_config.backend}")
