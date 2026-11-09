# GTEx Tissue Explorer desktop offline

## Menjalankan versi portable

Buka `dist/workspace_switch_20261030/GTExTissueExplorer/GTExTissueExplorer.exe` untuk build
launcher workspace 30 Oktober 2026. Jendela desktop menjalankan
antarmuka Streamlit dan model XGBoost secara lokal. Tidak perlu membuka browser,
menginstal Python, mengunduh GTEx, atau menjalankan notebook.

Untuk memindahkan aplikasi, salin **seluruh folder GTExTissueExplorer**, termasuk
folder `_internal`. Jangan memindahkan hanya berkas `.exe`. Semua tabel,
model final, data contoh, dan dependensi Python sudah disertakan.

Target: Windows 10/11 64-bit dengan Microsoft Edge WebView2 Runtime dan .NET
Framework yang tersedia. WebView2 telah terpasang di komputer pengembangan.
Jika runtime belum tersedia pada komputer tujuan, pasang menggunakan installer
offline resmi sebelum menjalankan aplikasi. Aplikasi tidak memasangnya atau
mengunduhnya otomatis. Runtime Windows ini tidak disertakan dalam folder portable.

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

## Menjalankan dari source

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\run_desktop_app.bat
```

Pemasangan dependensi untuk pengembangan dapat memerlukan internet. Ini terpisah
dari pemakaian executable. Launcher tidak menjalankan pip saat aplikasi dibuka.
Launcher browser `run_offline_app.bat` tetap tersedia sebagai alternatif.

## Pemeriksaan dan build

```powershell
.\run_desktop_app.ps1 -CheckOnly
.\build_desktop.ps1
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

`--check` juga tersedia pada executable. Pemeriksaan memvalidasi hash model,
memuat tabel dan aset, menjalankan audit dan inferensi data contoh, serta memeriksa
server dan halaman lokal dengan akses jaringan eksternal backend diblokir.
Server uji dihentikan setelah pemeriksaan. Ini tidak membuka jendela.

Build menggunakan versi Streamlit dan pywebview yang dipin agar API desktop
konsisten. Ketika model atau tabel penelitian berubah, jalankan build kembali.
Pengubahan kemasan desktop tidak mengubah selector, preprocessing, atau model.

Jika startup gagal, baca `last_error.txt` dan `desktop.log` pada direktori log.
Penyebab yang perlu diperiksa mencakup runtime WebView2 yang tidak tersedia,
berkas `_internal` yang tidak ikut disalin, atau bundle model yang rusak.
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

