"""Readable UI explanations for persisted errors, including older training runs."""
import re


FIELD_NAMES = {
    'top_k': 'jumlah gen (top-k)', 'C': 'C regularisasi L1',
    'max_iter': 'maksimum iterasi L1', 'tol': 'toleransi L1',
    'n_neighbors': 'jumlah tetangga MI', 'folds': 'jumlah fold', 'holdout_percent': 'porsi data holdout',
    'n_estimators': 'putaran boosting', 'learning_rate': 'learning rate',
    'max_depth': 'max depth', 'min_child_weight': 'min child weight',
    'subsample': 'subsample', 'colsample_bytree': 'colsample bytree',
    'reg_lambda': 'lambda XGBoost', 'reg_alpha': 'alpha XGBoost',
    'gamma': 'gamma', 'n_jobs': 'jumlah thread CPU',
}


def user_error(error) -> str:
    message = str(error)
    support = re.fullmatch(r'L1 has (\d+) nonzero genes, fewer than top-k (\d+)\. Lower k or adjust C\.', message)
    if support:
        found, requested = (f'{int(n):,}'.replace(',', '.') for n in support.groups())
        return (f'L1 menghasilkan {found} gen berkoefisien nonzero, kurang dari top-k {requested}. '
                'Kurangi jumlah gen atau naikkan C, lalu mulai pelatihan baru dengan konfigurasi tersebut.')
    if message.startswith('L1 did not converge'):
        return ('Regresi logistik L1 belum konvergen. Tambah maksimum iterasi atau sesuaikan toleransi, '
                'lalu mulai pelatihan baru. Model belum disimpan.')
    numeric = re.fullmatch(r'(\w+) must be (a finite number|an integer|between (.+) and (.+))', message)
    if numeric:
        field, condition, low, high = numeric.groups()
        label = FIELD_NAMES.get(field, field)
        if condition == 'a finite number':
            return f'Isi {label} dengan angka yang valid. Nilai kosong, NaN, dan tak hingga tidak dapat digunakan.'
        if condition == 'an integer':
            return f'Gunakan bilangan bulat untuk {label}.'
        return f'Nilai {label} harus berada antara {low} dan {high}.'
    missing = re.fullmatch(r'Fold (\d+) (training|validation) missing tissue classes: (.+)', message)
    if missing:
        fold, side, tissues = missing.groups()
        return (f'Data {side} pada fold {fold} tidak mencakup jaringan: {tissues}. '
                'Pilih jumlah fold lain. Pembagian donor dan kelas jaringan tidak diubah otomatis.')
    if 'hash' in message.lower():
        return ('Pemeriksaan integritas berkas model gagal karena hash tidak cocok. '
                'Pulihkan berkas asli atau latih dan simpan model baru. Model ini tidak digunakan.')
    if message in {'A training worker is already running', 'A training worker is starting'}:
        return 'Masih ada pelatihan yang berjalan atau sedang dimulai. Tunggu hingga selesai atau batalkan proses aktif.'
    if message.startswith('Worker stopped.'):
        return 'Proses pelatihan berhenti sebelum selesai. Lanjutkan proses tersimpan untuk memakai kembali tahap yang telah selesai.'
    return message
