# Quantum K-Means for Psychological Time Series Clustering

Tái hiện và mở rộng đề tài **feature-based clustering trên chuỗi thời gian tâm lý** (Bringmann et al., 2023) bằng cách so sánh Classical K-Means với **Quantum K-Means sử dụng Swap Test** chạy trên GPU simulator (Qiskit Aer).

---

## Tổng quan

Dữ liệu ESM (Experience Sampling Method) ghi lại trải nghiệm hàng ngày của **156 người di cư tại Hà Lan** qua 12 biến tâm lý (well-being, thái độ với nhóm chủ đạo, chất lượng tương tác...), đo nhiều lần/ngày trong vài tuần.

**Câu hỏi nghiên cứu**: Có những kiểu hành trình tâm lý (typology) nào trong quá trình hòa nhập văn hóa?

**Đóng góp mới** so với bài gốc: thay thế và so sánh Classical K-Means bằng **Quantum K-Means (Swap Test)** — lượng hóa bước tính khoảng cách bằng mạch lượng tử amplitude encoding + controlled-SWAP.

---

## Pipeline

```
Raw ESM data (S1 + S2 + S3)
        ↓ 01. Loading & Merge                             cleanM, reset TIDnum, gộp 3 study → dtAll (156 người)
        ↓ 02. Processing                                  GAM feature extraction (6 features × 12 items = 72 chiều)
                                                          Imputation (MICE) + Z-score + PCA → pca_scores (156 × 27)
        ↓ 03. Classical KMeans vs Quantum KMeans          Classical KMeans vs Quantum KMeans, k=2..10
        ↓ 04. Dims Comparison                             Quantum KMeans theo dims = 2, 5, 10, 15, 20, 27
```

### 6 đặc trưng động (dynamic features) trích từ mỗi chuỗi

| Feature | Ý nghĩa tâm lý |
|---|---|
| `median` | Mức trung bình (level) |
| `mad` | Dao động tổng thể (affective instability) |
| `mac` | Biến động từng bước (volatility) |
| `ar01` | Quán tính cảm xúc (emotional inertia) |
| `lin` | Xu hướng tuyến tính (adaptation trajectory) |
| `edf` | Độ phi tuyến của quỹ đạo (complexity) |

---

## Quantum K-Means — Swap Test

Giữ nguyên vòng lặp Lloyd's algorithm, **chỉ thay phép tính khoảng cách**:

1. **Amplitude Encoding**: vector 27 chiều → chuẩn hóa L2 → encode vào trạng thái lượng tử 5 qubit (`StatePreparation`)
2. **Swap Test**: đo fidelity |⟨ψ_x|ψ_c⟩|² giữa điểm dữ liệu và tâm cụm qua mạch Controlled-SWAP
3. **Distance** = `2 × (1 − P(|0⟩))`, ước lượng từ `shots=2048` lần đo
4. **Tối ưu**: pre-compile toàn bộ N circuit của X một lần duy nhất (cache), chỉ re-transpile K circuit tâm cụm mỗi iteration
5. **Checkpoint 3 cấp** (per-iteration / per-run / per-k): resume được khi bị ngắt

```bash
# Chạy QKM cho k=5
python qkm_runner.py 5 /path/to/pipeline_artifacts.pkl /path/to/checkpoints/
```

---

## Kết quả

### Classical vs Quantum (dims=27, k=2..10)

| | Classical KMeans | Quantum KMeans |
|---|---|---|
| Silhouette (TB) | **0.057** | 0.004 |
| Davies-Bouldin (TB) | **2.75** | 4.75 |
| Iterations | **4–18** (hội tụ) | **50/50** (không hội tụ) |
| ARI trung bình | — | 0.178 |
| Best k (Silhouette) | **k=2** | k=4 |

### QKM theo số chiều đầu vào

| dims | QKM Silhouette | ARI với Classical | Gap Silhouette |
|---|---|---|---|
| 2 | 0.010 | 0.068 | -0.067 |
| 10 | 0.009 | 0.127 | -0.083 |
| 15 | **0.018** | 0.176 | -0.054 |
| 27 | 0.018 | **0.183** | **-0.037** |

→ ARI tăng và gap thu hẹp khi dims tăng — tín hiệu tích cực nhưng QKM chưa vượt Classical ở bất kỳ cấu hình nào.

### Hạn chế chính của QKM

1. **Shot noise** (±2.2% với 2048 shots) gây dao động nhãn → không hội tụ ở 100% cấu hình
2. **L2 normalize** mất thông tin magnitude → người "ổn định" và "bất ổn" nhưng cùng "hướng" bị coi là giống nhau
3. **Cosine distance ≠ Euclidean distance** → 2 phương pháp đo "giống nhau" theo 2 định nghĩa hình học khác nhau

---

## Cài đặt

```bash
pip install qiskit==1.4.2 qiskit-aer==0.15.1 qiskit-machine-learning==0.7.2
pip install numpy pandas scikit-learn pygam rpy2 pyreadr
pip install tqdm plotly great_tables
```

> GPU (NVIDIA): cài thêm `qiskit-aer-gpu` thay cho `qiskit-aer`

```bash
pip install qiskit-aer-gpu==0.7.2
```

---

## Cấu trúc repo

```
├── 01_Loading_and_Merge_data.ipynb          # cleanM, gộp S1+S2+S3 → dtAll
├── 02_Processing_Psychological_Time_Series.ipynb  # feature extraction, PCA → artifacts.pkl
├── 03_Analysis_Only.ipynb                   # Classical vs QKM, k=2..10
├── 04_Dims_Comparison_v2.ipynb              # QKM theo dims=2,5,10,15,20,27
├── qkm_runner.py                            # script chạy QKM từ command line (GCP/VM)
├── results/
│   ├── clustering_metrics.csv
│   ├── dims_comparison_metrics.csv
│   └── classical_vs_qkm_by_dims.csv
└── data/
    └── README.md                            # hướng dẫn download dữ liệu từ OSF
```

### Thứ tự chạy

```
01 → 02 → (qkm_runner.py hoặc 03) → 04
```

Notebook `02` xuất `pipeline_artifacts.pkl` — đây là đầu vào bắt buộc cho `03`, `04`, và `qkm_runner.py`.

---

## Dữ liệu

Dữ liệu **không được lưu trong repo** vì lý do quyền riêng tư người tham gia nghiên cứu.

Download `osf_mini.Rda` và `osf_var_meta.xlsx` tại:
> **OSF**: https://doi.org/10.17605/OSF.IO/J8DZV

Sau đó cập nhật đường dẫn ở đầu notebook `01` và `02`.

---

## Citation

Nếu bạn dùng code này, vui lòng cite:

```bibtex
@article{bringmann2023,
  title   = {A Gentle Introduction and Application of Feature-Based 
             Clustering with Psychological Time Series},
  author  = {Bringmann, L. F. and others},
  year    = {2023},
  url     = {https://www.tsfeatureclustr.com/}
}

@misc{kreienkamp2023,
  title   = {Psychological time series data — migration/cultural adaptation},
  author  = {Kreienkamp, J. and others},
  year    = {2023},
  doi     = {10.17605/OSF.IO/J8DZV}
}
```
### Bài báo gốc (methodology)
Kreienkamp, J., Agostini, M., Monden, R., Epstude, K., de Jonge, P., & Bringmann, L. F. (2024).
A Gentle Introduction and Application of Feature-Based Clustering with Psychological Time Series.
*Multivariate Behavioral Research*, 60(2), 362–392.
https://doi.org/10.1080/00273171.2024.2432918

### Tutorial & code gốc
https://www.tsfeatureclustr.com/

### Dataset
Kreienkamp et al. (2023). OSF.
https://doi.org/10.17605/OSF.IO/J8DZV

**Dependencies:**
- [Qiskit](https://github.com/Qiskit/qiskit) — Apache License 2.0
- [Qiskit Aer](https://github.com/Qiskit/qiskit-aer) — Apache License 2.0

---

## License

MIT License — xem file [LICENSE](LICENSE)
