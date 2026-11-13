# GTEx Tissue Explorer desktop offline

## Menjalankan versi desktop

Jalankan dari folder aplikasi, dengan `.venv` yang memuat `requirements-desktop.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\.venv\Scripts\python.exe desktop.py
```

Jendela desktop menjalankan antarmuka Streamlit dan model XGBoost secara lokal,
tanpa membuka browser, mengunduh GTEx, atau menjalankan notebook. Pemasangan
dependensi pertama dapat memerlukan internet. `desktop.py` tidak menjalankan pip
saat aplikasi dibuka. Untuk mode browser, lihat bagian Menjalankan aplikasi pada
[README](../README.md).

Executable portable (`GTExTissueExplorer.exe`) tidak tersedia di repository ini.
Build lama dihapus pada 11 November 2026 karena tidak memuat perubahan terbaru.
Skrip build (`build_desktop.ps1`) dan spesifikasi PyInstaller tidak tersimpan,
sehingga executable baru memerlukan skrip build PyInstaller yang disiapkan ulang
dari `desktop.py`. PyInstaller sudah dipin di `requirements-desktop.txt`.

Target: Windows 10/11 64-bit dengan Microsoft Edge WebView2 Runtime dan .NET
Framework yang tersedia. WebView2 telah terpasang di komputer pengembangan.
Jika runtime belum tersedia, pasang menggunakan installer offline resmi sebelum
menjalankan aplikasi. Aplikasi tidak memasangnya atau mengunduhnya otomatis.

## Apa yang dimaksud offline

- Tidak memerlukan koneksi internet untuk audit, inferensi, tabel, grafik,
  pelatihan lokal, atau ekspor hasil.
- Backend hanya mendengarkan `127.0.0.1` pada port kosong yang dipilih otomatis.
- Koneksi Python ke alamat non-loopback dan pencarian DNS eksternal ditolak.
- Content Security Policy membatasi sumber antarmuka ke sumber lokal.
- Telemetri Streamlit dimatikan. Tidak ada akun, API, atau layanan cloud.
- Pelatihan model baru memakai data development lokal. Model hasil pelatihan
  disimpan terpisah dari model dan hasil eksperimen notebook.
- Menutup jendela menghentikan server lokal yang dimulai oleh jendela tersebut.

HTTP/WebSocket localhost tetap digunakan untuk menghubungkan jendela dan backend.
Ini komunikasi dalam komputer yang sama, bukan akses internet. Pembatasan di atas
berlaku pada aplikasi, bukan firewall sistem atau aktivitas pembaruan Windows.

Unggahan diproses dalam memori aplikasi. CSV hasil disimpan ke lokasi pilihan
pengguna melalui dialog unduh. Log dan profil WebView berada di
`%LOCALAPPDATA%/GTExTissueExplorer`. Log digunakan untuk diagnostik startup,
bukan untuk menyimpan matriks unggahan.

## Pemeriksaan

```powershell
.\.venv\Scripts\python.exe desktop.py --check
.\.venv\Scripts\python.exe desktop.py --check-training
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

`--check` memvalidasi hash model, memuat tabel dan aset, menjalankan audit dan
inferensi data contoh, serta memeriksa server dan halaman lokal dengan akses
jaringan eksternal backend diblokir. Server uji dihentikan setelah pemeriksaan.
Ini tidak membuka jendela. `--check-training` memeriksa worker dengan fit ANOVA
singkat.

Versi Streamlit dan pywebview dipin di `requirements-desktop.txt` agar API desktop
konsisten.

Jika startup gagal, baca `last_error.txt` dan `desktop.log` pada direktori log.
Penyebab yang perlu diperiksa mencakup runtime WebView2 yang tidak tersedia atau
bundle model yang rusak.
# Pelatihan model

Layar **Pilih workspace** tampil sebelum antarmuka utama. Cari atau pilih
workspace, lalu tekan **Buka workspace** untuk masuk ke **Latih Model**.
Dataset, proses, checkpoint, model, dan gallery pengguna dipisahkan per workspace.
Saat memilih **Buat workspace**, centang **Sertakan model
bawaan** untuk menyediakan model asli dan baseline, atau kosongkan untuk
memulai tanpa model prediksi. File lama tetap berada di **Default**. Workspace
baru memakai subfolder `workspaces/ws_<uuid>/` di direktori state. Mengubah nama
tidak memindahkan file. **Ganti** pada blok workspace di bagian bawah sidebar
kembali ke layar pemilihan.
Memilih baris belum mengganti project aktif sampai **Buka workspace** ditekan.
Berpindah workspace tidak menghentikan worker aktif,
dan **Buka proses** membuka workspace yang menjalankannya. Satu worker training
berlaku untuk seluruh workspace. Aset referensi eksperimen tetap bersama.

Gunakan Pelatihan Pengguna > Latih Model untuk menyimpan konfigurasi baru. Pilih model tersimpan
di Prediksi Jaringan, lalu audit ulang matriks input. File model asli dan tabel
eksperimen tidak ditimpa. Data training disertakan sebagai ekspor development,
tanpa holdout dan tanpa memerlukan dataset GCT asli pada komputer pengguna.

Worker dan hasil disimpan secara lokal pada direktori state aplikasi. Menutup
halaman tidak memulai ulang fit. **Pause** mempertahankan proses di memori, lalu
**Resume** meneruskan fit yang sama. Worker harus tetap hidup, sehingga jeda
tidak bertahan setelah komputer dimulai ulang. **Stop** menghentikan worker
aktif. Fold yang selesai tetap tersimpan. Saat dilanjutkan, fit yang terputus
diulang melalui **Continue from checkpoint** dan model hanya diterbitkan setelah seluruh training selesai.
Gunakan `--check-training` untuk pemeriksaan worker offline dengan model uji kecil.

