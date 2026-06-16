"""
====================================================================
  Prediksi Tingkat Kemiskinan Indonesia - Model KNN
  Dataset: 514 Kabupaten/Kota di seluruh Indonesia
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  PREDIKSI KEMISKINAN INDONESIA - K-NEAREST NEIGHBORS")
print("=" * 60)

# Opsi 2 - pakai slash biasa
df = pd.read_csv("Klasifikasi_Tingkat_Kemiskinan_Indonesia.csv")
print(f"\n✔ Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

# Kolom fitur & target
FITUR = [
    "Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)",
    "Rata-rata Lama Sekolah Penduduk 15+ (Tahun)",
    "Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)",
    "Indeks Pembangunan Manusia",
    "Umur Harapan Hidup (Tahun)",
    "Persentase rumah tangga yang memiliki akses terhadap sanitasi layak",
    "Persentase rumah tangga yang memiliki akses terhadap air minum layak",
    "Tingkat Pengangguran Terbuka",
    "Tingkat Partisipasi Angkatan Kerja",
    "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)",
]
TARGET = "Klasifikasi Kemiskinan"

# ─────────────────────────────────────────────
# 2. PRA-PROSES
# ─────────────────────────────────────────────
# Konversi kolom bertipe str (koma desimal) ke float
for col in FITUR:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", "."),
            errors="coerce"
        )

# Tangani nilai hilang
missing = df[FITUR + [TARGET]].isnull().sum().sum()
if missing > 0:
    for col in FITUR:
        df[col] = df[col].fillna(df[col].median())
    print(f"⚠  {missing} nilai hilang diisi dengan median.")
else:
    print("✔ Tidak ada nilai hilang.")

X = df[FITUR].to_numpy(dtype=float)
y = df[TARGET].values.astype(int)

print(f"\nDistribusi kelas:")
vals, cnts = np.unique(y, return_counts=True)
label_map = {0: "Tidak Miskin", 1: "Miskin"}
for v, c in zip(vals, cnts):
    print(f"  Kelas {v} ({label_map[v]}): {c} data ({c/len(y)*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. SPLIT DATA
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nPembagian data:")
print(f"  Data latih : {X_train.shape[0]} sampel")
print(f"  Data uji   : {X_test.shape[0]} sampel")

# ─────────────────────────────────────────────
# 4. NORMALISASI
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 5. CARI K OPTIMAL
# ─────────────────────────────────────────────
print("\n─ Mencari nilai K optimal (1–20) ─")
k_range = range(1, 21)
cv_scores = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    score = cross_val_score(knn, X_train_sc, y_train, cv=cv, scoring="accuracy")
    cv_scores.append(score.mean())

best_k = k_range[np.argmax(cv_scores)]
best_cv_acc = max(cv_scores)
print(f"  K terbaik : {best_k}  (CV accuracy = {best_cv_acc:.4f})")

# ─────────────────────────────────────────────
# 6. LATIH MODEL FINAL
# ─────────────────────────────────────────────
model = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]

acc  = accuracy_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)

print(f"\n{'='*60}")
print(f"  HASIL EVALUASI MODEL KNN (K={best_k})")
print(f"{'='*60}")
print(f"  Akurasi        : {acc*100:.2f}%")
print(f"  AUC-ROC        : {auc:.4f}")
print(f"\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred,
      target_names=["Tidak Miskin", "Miskin"]))

# ─────────────────────────────────────────────
# 7. VISUALISASI
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Prediksi Tingkat Kemiskinan Indonesia — KNN", fontsize=14, fontweight="bold")

# (a) K vs Akurasi
axes[0].plot(list(k_range), cv_scores, marker="o", color="#2563eb", linewidth=2)
axes[0].axvline(best_k, color="#dc2626", linestyle="--", label=f"K optimal = {best_k}")
axes[0].set_xlabel("Nilai K")
axes[0].set_ylabel("Akurasi (5-Fold CV)")
axes[0].set_title("Pencarian K Optimal")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# (b) Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Tidak Miskin", "Miskin"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Confusion Matrix")

# (c) ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[2].plot(fpr, tpr, color="#16a34a", linewidth=2, label=f"AUC = {auc:.4f}")
axes[2].plot([0,1],[0,1],"--", color="gray")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].set_title("ROC Curve")
axes[2].legend(loc="lower right")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("hasil_knn_kemiskinan.png", dpi=150, bbox_inches="tight")
print("\n✔ Grafik disimpan: hasil_knn_kemiskinan.png")

# ─────────────────────────────────────────────
# 8. PREDIKSI CONTOH BARU
# ─────────────────────────────────────────────
print("\n─ Contoh Prediksi Data Baru ─")
contoh = pd.DataFrame([{
    "Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)": 22.5,
    "Rata-rata Lama Sekolah Penduduk 15+ (Tahun)": 7.2,
    "Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)": 7000,
    "Indeks Pembangunan Manusia": 62.0,
    "Umur Harapan Hidup (Tahun)": 63.5,
    "Persentase rumah tangga yang memiliki akses terhadap sanitasi layak": 55.0,
    "Persentase rumah tangga yang memiliki akses terhadap air minum layak": 70.0,
    "Tingkat Pengangguran Terbuka": 6.0,
    "Tingkat Partisipasi Angkatan Kerja": 65.0,
    "PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)": 2000000,
}])[FITUR]

contoh_sc  = scaler.transform(contoh.values)
pred_class = model.predict(contoh_sc)[0]
pred_prob  = model.predict_proba(contoh_sc)[0]

print(f"  Prediksi kelas : {pred_class} → {label_map[pred_class]}")
print(f"  Probabilitas   : Tidak Miskin={pred_prob[0]*100:.1f}%  |  Miskin={pred_prob[1]*100:.1f}%")

print("\n✔ Selesai!")
