"""Readable UI explanations for persisted errors, including older training runs."""

import re

from gtex_explorer.display import count
from gtex_explorer.training.run_labels import XGBOOST_LABELS

FIELD_NAMES = {
    # The field labels of the Latih Model form, so an error names the field to fix.
    "top_k": "Top-k gen",
    "C": "C (regularisasi)",
    "max_iter": "Maksimum iterasi",
    "tol": "Toleransi",
    "n_neighbors": "n_neighbors",
    "folds": "Jumlah fold",
    "holdout_percent": "Porsi data holdout",
    **{name: label for name, label in XGBOOST_LABELS.items()},
}


def user_error(error) -> str:
    """Translate a technical error into a message for the user."""
    message = str(error)
    support = re.fullmatch(r"L1 has (\d+) nonzero genes, fewer than top-k (\d+)\. Lower k or adjust C\.", message)
    if support:
        found, requested = (count(n) for n in support.groups())
        return (
            f"L1 menghasilkan {found} gen berkoefisien nonzero, kurang dari top-k {requested}. "
            "Kurangi Top-k gen atau naikkan C (regularisasi), lalu mulai pelatihan baru dengan konfigurasi tersebut."
        )
    if message.startswith("L1 did not converge"):
        return (
            "Regresi logistik L1 belum konvergen. Tambah maksimum iterasi atau sesuaikan toleransi, "
            "lalu mulai pelatihan baru. Model belum disimpan."
        )
    numeric = re.fullmatch(r"(\w+) must be (a finite number|an integer|between (.+) and (.+))", message)
    if numeric:
        field, condition, low, high = numeric.groups()
        label = FIELD_NAMES.get(field, field)
        if condition == "a finite number":
            return f"Isi {label} dengan angka yang valid. Nilai kosong, NaN, dan tak hingga tidak dapat digunakan."
        if condition == "an integer":
            return f"Gunakan bilangan bulat untuk {label}."
        return f"Nilai {label} harus berada antara {low} dan {high}."
    missing = re.fullmatch(r"Fold (\d+) (training|validation) missing tissue classes: (.+)", message)
    if missing:
        fold, side, tissues = missing.groups()
        return (
            f"Data {side} pada fold {fold} tidak mencakup jaringan: {tissues}. "
            "Pilih jumlah fold lain. Pembagian donor dan kelas jaringan tidak diubah otomatis."
        )
    if re.fullmatch(r"Fold \d+ is empty", message) or "n_splits=" in message:
        return (
            "Sampel atau donor pada data development terlalu sedikit untuk jumlah fold ini. "
            "Kurangi jumlah fold, kurangi porsi data holdout, atau nonaktifkan cross-validation."
        )
    if "hash" in message.lower():
        return (
            "Pemeriksaan integritas berkas model gagal karena hash tidak cocok. "
            "Pulihkan berkas asli atau latih dan simpan model baru. Model ini tidak digunakan."
        )
    if message in {"A training worker is already running", "A training worker is starting"}:
        return (
            "Masih ada pelatihan yang berjalan atau sedang dimulai. Tunggu hingga selesai atau batalkan proses aktif."
        )
    if message.startswith("Worker stopped."):
        return "Proses pelatihan berhenti sebelum selesai. Lanjutkan proses tersimpan untuk memakai kembali tahap yang telah selesai."
    return message
