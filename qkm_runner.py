import sys
import subprocess
import warnings
import os
import pickle
import threading
import time

import numpy as np
warnings.filterwarnings('ignore')


# ==============================================================================
# Đọc tham số k từ command line
# ==============================================================================

if len(sys.argv) < 2:
    sys.exit(1)

K_TARGET = int(sys.argv[1])
print(f"K_TARGET = {K_TARGET}")


# ==============================================================================
# Setup GPU + AerSimulator
# ==============================================================================

try:
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    ).decode().strip()
    print(f"GPU detected: {gpu_info}")
    HAS_GPU = True
except Exception:
    print("Khong phat hien GPU — chay CPU fallback.")
    HAS_GPU = False

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library import StatePreparation  # ✅ nhanh hơn initialize()
from tqdm.auto import tqdm as _tqdm

if HAS_GPU:
    SIM = AerSimulator(method='statevector', device='GPU')
    DEVICE_LABEL = "GPU (statevector)"
else:
    SIM = AerSimulator(method='statevector', device='CPU')
    DEVICE_LABEL = "CPU (statevector)"

# Smoke test
_qc = QuantumCircuit(2)
_qc.h(0); _qc.cx(0, 1); _qc.measure_all()
_result = SIM.run(_qc, shots=128).result()
print(f"AerSimulator [{DEVICE_LABEL}] OK — test: {dict(_result.get_counts())}")

SHOTS = 2048


# ==============================================================================
# Paths
# ==============================================================================

ARTIFACT_PATH  = sys.argv[2] if len(sys.argv) > 2 \
                 else '/content/drive/MyDrive/pipeline_B_artifacts.pkl'
CHECKPOINT_DIR = sys.argv[3] if len(sys.argv) > 3 \
                 else '/content/drive/MyDrive/qkm_checkpoints'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"ARTIFACT_PATH  = {ARTIFACT_PATH}")
print(f"CHECKPOINT_DIR = {CHECKPOINT_DIR}")

with open(ARTIFACT_PATH, 'rb') as f:
    artifacts = pickle.load(f)

import pandas as pd
from sklearn.preprocessing import normalize

pca_scores = artifacts['pca_scores']
pca_cutoff = artifacts['pca_cutoff']
X_full     = pca_scores.values.astype(float)
X          = X_full[:, :pca_cutoff]

print(f"Data: {X.shape} | pca_cutoff={pca_cutoff}")


# ==============================================================================
# Checkpoint helpers
# ==============================================================================

def save_ckpt(name: str, obj) -> None:
    with open(f"{CHECKPOINT_DIR}/{name}.pkl", 'wb') as f:
        pickle.dump(obj, f)
    print(f"  >> Saved  : {name}")


def load_ckpt(name: str):
    path = f"{CHECKPOINT_DIR}/{name}.pkl"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            print(f"  >> Loaded : {name}")
            return pickle.load(f)
    return None


def delete_ckpt(name: str) -> None:
    path = f"{CHECKPOINT_DIR}/{name}.pkl"
    if os.path.exists(path):
        os.remove(path)
        print(f"  >> Deleted: {name}")


# ==============================================================================
# Các hàm lượng tử (đã tối ưu)
# ==============================================================================

def _pad_normalize(vec: np.ndarray):
    """Pad vector lên 2^n và normalize. Trả về (vec_normalized, n_qubits)."""
    vec  = np.array(vec, dtype=float)
    n    = max(int(np.ceil(np.log2(len(vec)))), 1)
    size = 2 ** n
    if len(vec) < size:
        vec = np.pad(vec, (0, size - len(vec)))
    norm = np.linalg.norm(vec)
    if norm > 1e-10:
        vec = vec / norm
    else:
        vec    = np.zeros(size)
        vec[0] = 1.0
    return vec, n


def amplitude_encode(vec: np.ndarray) -> QuantumCircuit:
    """
    Tạo QuantumCircuit n qubit dùng StatePreparation (nhanh hơn initialize).
    """
    vec, n = _pad_normalize(vec)
    qc = QuantumCircuit(n)
    qc.append(StatePreparation(vec.tolist()), list(range(n)))
    return qc


def build_swap_test_circuit(qc_x: QuantumCircuit,
                             qc_y: QuantumCircuit) -> QuantumCircuit:
    """
    Swap Test giữa 2 QuantumCircuit đã encode sẵn.
    Nhận circuit thay vì vector để tái sử dụng qc_x (cache).
    """
    n_q  = qc_x.num_qubits
    anc  = QuantumRegister(1,   'anc')
    rx   = QuantumRegister(n_q, 'rx')
    ry   = QuantumRegister(n_q, 'ry')
    creg = ClassicalRegister(1, 'c')
    qc   = QuantumCircuit(anc, rx, ry, creg)

    qc.compose(qc_x, qubits=rx, inplace=True)
    qc.compose(qc_y, qubits=ry, inplace=True)
    qc.h(anc[0])
    for i in range(n_q):
        qc.cswap(anc[0], rx[i], ry[i])
    qc.h(anc[0])
    qc.measure(anc[0], creg[0])
    return qc


def _gpu_spinner(label: str, stop_event: threading.Event) -> None:
    t0 = time.time()
    while not stop_event.is_set():
        elapsed = time.time() - t0
        print(f"\r    GPU running: {label} ... {elapsed:.1f}s",
              end="", flush=True)
        time.sleep(0.5)
    elapsed = time.time() - t0
    print(f"\r    GPU done:  {label} ({elapsed:.1f}s)          ", flush=True)


def precompile_x_circuits(X: np.ndarray, simulator: AerSimulator) -> list:
    """
    KEY OPTIMIZATION: Encode + transpile tất cả X circuits 1 lần duy nhất.
    X không thay đổi giữa các iter/run → cache lại dùng mãi.
    """
    print(f"  Pre-compiling {len(X)} X circuits (1 lần duy nhất)...")
    t0 = time.time()
    raw = [amplitude_encode(X[i]) for i in range(len(X))]
    # Transpile batch 1 lần
    compiled = transpile(raw, simulator, optimization_level=1)
    print(f"  Pre-compile done: {time.time()-t0:.1f}s")
    return compiled


def swap_test_distance_matrix_gpu(
    X_compiled: list,
    X: np.ndarray,
    centers: np.ndarray,
    simulator: AerSimulator,
    shots: int = SHOTS,
    pbar: _tqdm = None
) -> np.ndarray:
    """
    Tính ma trận khoảng cách N x K.
    X_compiled: list circuits của X đã transpile sẵn (cache từ ngoài).
    centers: transpile mỗi iter vì thay đổi.
    """
    N, K           = len(X), len(centers)
    total_circuits = N * K
    label          = f"{total_circuits:,} circuits x {shots:,} shots"

    # Encode + transpile centers (chỉ K circuits, nhỏ)
    center_circuits = transpile(
        [amplitude_encode(centers[k]) for k in range(K)],
        simulator, optimization_level=1
    )

    # Build swap test circuits dùng X đã cache
    circuits = [
        build_swap_test_circuit(X_compiled[i], center_circuits[k])
        for i in range(N) for k in range(K)
    ]

    # Transpile toàn bộ batch swap test
    t_compile = time.time()
    circuits = transpile(circuits, simulator, optimization_level=1)
    print(f"    transpile batch: {time.time()-t_compile:.1f}s", flush=True)

    stop_evt = threading.Event()
    spinner  = threading.Thread(
        target=_gpu_spinner, args=(label, stop_evt), daemon=True
    )
    spinner.start()

    job    = simulator.run(circuits, shots=shots)
    result = job.result()

    stop_evt.set()
    spinner.join()

    if pbar is not None:
        pbar.set_postfix_str(f"last batch: {label}")

    dist_mat = np.zeros((N, K))
    for idx, (i, k) in enumerate(
        [(i, k) for i in range(N) for k in range(K)]
    ):
        counts         = result.get_counts(idx)
        p0             = counts.get('0', 0) / shots
        dist_mat[i, k] = 2.0 * (1.0 - p0)

    return dist_mat


# ==============================================================================
# QiskitSwapTestKMeans (đã tối ưu)
# ==============================================================================

class QiskitSwapTestKMeans:
    """
    Quantum K-Means với Swap Test — đã tối ưu:
      - Cache X circuits (pre-compile 1 lần)
      - Chỉ transpile K center circuits mỗi iter
      - Tất cả checkpoint logic giữ nguyên
    """

    def __init__(self,
                 n_clusters:   int          = 3,
                 n_init:       int          = 5,
                 max_iter:     int          = 50,
                 shots:        int          = SHOTS,
                 simulator:    AerSimulator = None,
                 random_state: int          = 42,
                 verbose:      bool         = False):
        self.n_clusters   = n_clusters
        self.n_init       = n_init
        self.max_iter     = max_iter
        self.shots        = shots
        self.simulator    = simulator or SIM
        self.random_state = random_state
        self.verbose      = verbose

        self.labels_              = None
        self.cluster_centers_     = None
        self.inertia_             = None
        self.best_run_history_    = []
        self.convergence_history_ = []

        # Cache X circuits — set trong fit()
        self._X_compiled = None

    def _euclidean_inertia(self,
                            X: np.ndarray,
                            labels: np.ndarray,
                            centers: np.ndarray) -> float:
        return float(sum(
            np.sum((X[labels == k] - centers[k]) ** 2)
            for k in range(self.n_clusters)
        ))

    def _run_once(self,
                  X: np.ndarray,
                  seed: int,
                  k_label: str = None,
                  run_idx: int = None):
        rng     = np.random.RandomState(seed)
        n       = X.shape[0]
        N, K    = n, self.n_clusters
        centers = X[rng.choice(n, K, replace=False)].copy()
        labels  = np.zeros(n, dtype=int)
        history = []

        circuits_per_iter = N * K
        total_shots_fired = 0
        start_iter        = 0

        if k_label is not None:
            iter_ckpt = load_ckpt(f"{k_label}_run{run_idx}_iter")
            if iter_ckpt is not None:
                start_iter        = iter_ckpt['it'] + 1
                labels            = iter_ckpt['labels']
                centers           = iter_ckpt['centers']
                history           = iter_ckpt['history']
                total_shots_fired = iter_ckpt['total_shots_fired']
                print(f"    Resume iter: bat dau tu it={start_iter}")

        pbar = _tqdm(
            total      = self.max_iter,
            desc       = f"  k={K}",
            unit       = "iter",
            leave      = False,
            initial    = start_iter,
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} iter "
                         "[{elapsed}<{remaining}] {postfix}",
        )

        for it in range(start_iter, self.max_iter):
            shots_this_iter    = circuits_per_iter * self.shots
            total_shots_fired += shots_this_iter

            pbar.set_postfix({
                "iter"       : f"{it+1}/{self.max_iter}",
                "shots/iter" : f"{shots_this_iter:,}",
                "total_shots": f"{total_shots_fired:,}",
            })

            # Truyền X_compiled đã cache vào
            dist_mat   = swap_test_distance_matrix_gpu(
                self._X_compiled, X, centers,
                self.simulator, self.shots, pbar=pbar
            )
            new_labels = np.argmin(dist_mat, axis=1)
            inertia    = self._euclidean_inertia(X, new_labels, centers)
            history.append(inertia)
            pbar.update(1)

            if k_label is not None:
                save_ckpt(f"{k_label}_run{run_idx}_iter", {
                    'it'               : it,
                    'labels'           : new_labels,
                    'centers'          : centers,
                    'history'          : history,
                    'total_shots_fired': total_shots_fired,
                })

            if np.all(new_labels == labels) and it > 0:
                pbar.set_postfix_str(
                    f"converged iter {it+1} | "
                    f"total shots = {total_shots_fired:,}"
                )
                labels = new_labels
                break
            labels = new_labels

            for k in range(K):
                mask       = labels == k
                centers[k] = X[mask].mean(axis=0) if mask.sum() > 0 \
                             else X[rng.choice(n)]

        pbar.close()
        print(
            f"    {len(history)} iters | "
            f"{circuits_per_iter:,} circuits/iter | "
            f"{self.shots:,} shots/circuit | "
            f"total shots fired = {total_shots_fired:,}"
        )

        if k_label is not None:
            delete_ckpt(f"{k_label}_run{run_idx}_iter")

        final_inertia = self._euclidean_inertia(X, labels, centers)
        return labels, centers, final_inertia, history

    def fit(self, X, k_label: str = None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = normalize(X.astype(float), norm='l2')

        # Pre-compile X circuits 1 lần duy nhất cho toàn bộ fit()
        self._X_compiled = precompile_x_circuits(X, self.simulator)

        rng   = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 10_000, self.n_init)
        best  = np.inf
        self.convergence_history_ = []

        start_run = 0
        if k_label is not None:
            partial = load_ckpt(f"{k_label}_partial")
            if partial is not None:
                best                      = partial['best_inertia']
                self.labels_              = partial['best_labels']
                self.cluster_centers_     = partial['best_centers']
                self.best_run_history_    = partial['best_run_history']
                self.convergence_history_ = partial['convergence_history']
                start_run                 = partial['next_run']
                print(f"  Resume {k_label}: bat dau tu run {start_run+1}/{self.n_init}")

        for run_idx, seed in enumerate(seeds):
            if run_idx < start_run:
                continue

            print(f"  [QKM k={self.n_clusters}] "
                  f"Run {run_idx+1}/{self.n_init} (seed={seed})")

            lbl, ctr, iner, hist = self._run_once(
                X, seed, k_label=k_label, run_idx=run_idx
            )
            self.convergence_history_.append(hist)
            print(f"    inertia = {iner:.4f} | iters = {len(hist)}")

            if iner < best:
                best                   = iner
                self.labels_           = lbl.copy()
                self.cluster_centers_  = ctr.copy()
                self.best_run_history_ = hist

            if k_label is not None:
                save_ckpt(f"{k_label}_partial", {
                    'best_inertia'       : best,
                    'best_labels'        : self.labels_,
                    'best_centers'       : self.cluster_centers_,
                    'best_run_history'   : self.best_run_history_,
                    'convergence_history': self.convergence_history_,
                    'next_run'           : run_idx + 1,
                })

        self.inertia_ = best

        if k_label is not None:
            delete_ckpt(f"{k_label}_partial")
            save_ckpt(k_label, self)
            print(f"  Hoan tat {k_label} | best inertia = {best:.4f}")

        return self


# ==============================================================================
# Chạy k_target
# ==============================================================================

print("\n" + "=" * 55)
print(f"  Quantum K-Means — k = {K_TARGET}")
print("=" * 55)

ckpt = load_ckpt(f"quantum_k{K_TARGET}")
if ckpt is not None:
    print(f"  quantum_k{K_TARGET} da co checkpoint, skip.")
    print(f"  best inertia = {ckpt.inertia_:.4f}")
    sys.exit(0)

m = QiskitSwapTestKMeans(
    n_clusters   = K_TARGET,
    n_init       = 10,
    max_iter     = 50,
    shots        = SHOTS,
    simulator    = SIM,
    random_state = 123,
    verbose      = True,
).fit(X, k_label=f"quantum_k{K_TARGET}")

print(f"\n  XONG k={K_TARGET} | best inertia = {m.inertia_:.4f}")