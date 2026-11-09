# GTEx Tissue Classifier

Aplikasi Streamlit lokal untuk menampilkan hasil eksperimen, menjelajahi panel gen
terpilih, dan memprediksi kelas jaringan dari matriks TPM menggunakan bundle
XGBoost final. Seluruh pemrosesan, model, tabel, dan aset berjalan dari komputer
lokal tanpa API atau layanan web.

## Pelatihan model lokal

### Workspace

Saat aplikasi dibuka, layar **Pilih workspace** tampil sebelum antarmuka utama.
Cari atau pilih workspace, lalu tekan **Buka workspace** untuk masuk ke
**Latih Model**. Daftar dapat digulir, nama panjang dipersingkat di baris dengan
nama lengkap pada tooltip dan ringkasan pilihan. Tombol pembuka tetap berada
di luar area daftar. **Buat workspace** menyediakan pilihan
**Sertakan model bawaan**. Jika dicentang, model asli L1 dan baseline tersedia
bersama model pengguna. Jika tidak, workspace dimulai tanpa model prediksi dan
pengguna perlu melatih modelnya terlebih dahulu. Model bawaan dibaca dari aset
aplikasi tanpa disalin ke setiap workspace.

Hasil lama tetap tersedia di **Default** pada lokasi sebelumnya. Workspace baru
disimpan di `workspaces/ws_<uuid>/` pada direktori state aplikasi. **Ubah nama**
tidak mengubah lokasi file. Di dalam aplikasi, **Ganti** pada blok workspace
di atas **Referensi Eksperimen** kembali ke
layar pemilihan. Memilih baris, mencari, membuat, atau mengubah nama belum
mengganti workspace aktif. Pergantian melalui **Buka workspace** membersihkan
unggahan dan hasil audit/prediksi sebelumnya, tetapi tidak menghapus file tersimpan atau menghentikan
training. Tautan **Buka proses** membawa pengguna ke workspace proses aktif.
Hanya satu worker training dapat berjalan atau dijeda dalam seluruh aplikasi.

**Hapus** pada layar pemilihan menghapus permanen workspace beserta hasil
pelatihan, model, dan dataset unggahannya setelah namanya diketik persis. Workspace
**Default** dan workspace dengan pelatihan aktif tidak dapat dihapus. **Hapus model**
dan **Hapus hasil pelatihan** pada halaman Hasil Pelatihan menghapus satu run beserta
modelnya; dataset unggahan tetap tersimpan sampai workspace dihapus.
Referensi eksperimen, kamus, dan contoh lokal tetap merupakan aset bersama.
Nilai widget sementara dapat kembali ke default ketika halamannya tidak lagi
ditampilkan. Pemeriksaan integritas pada launcher desktop tetap memeriksa aset
bawaan sebelum membuka jendela, terpisah dari pemuatan halaman utama setelah entry.

**Latih Model** dan **Prediksi Jaringan** berada berurutan di bagian atas sidebar.
Kelompok **Gallery Hasil Pelatihan** memuat **Hasil Pelatihan** dan **Panel Gen**,
terpisah dari **Referensi Eksperimen** penelitian. Unggah matriks TPM serta
metadata yang memuat ID sampel, donor, dan kelas jaringan pada tab **Data training**.
Kolom metadata dapat dipilih. Sampel dicocokkan berdasarkan ID yang sama persis.
Gen dan kelas jaringan pengguna tidak harus sama dengan model eksperimen asli.
Atur ANOVA, Mutual Information, regresi logistik L1, atau baseline semua kandidat
pada langkah **Konfigurasi**. Periksa ringkasan pada langkah **Tinjau dan latih**
sebelum memilih **Latih dan simpan model**. Tombol Kembali mempertahankan pengaturan.
Jumlah kandidat (maksimum 5.000), top-k, parameter selector,
dan XGBoost dapat diubah. L1 harus konvergen dan memiliki cukup gen nonzero.

Training-only memakai seluruh dataset yang dipilih tanpa menghasilkan skor validation.
CV bersifat opsional, dengan 2-10 fold dan pemisahan donor. Cakupan kelas diperiksa
sebelum fitting. Imputasi, filter teknis, pemilihan kandidat berdasarkan varians,
standardisasi, serta seleksi supervised dipelajari terpisah pada data training
setiap fold. Model final menggunakan seluruh dataset training ini. Opsi **Data
eksperimen notebook** secara eksplisit memilih 807 sampel development tersimpan.
Holdout eksperimen asli tidak dipakai dan hasil aslinya tidak berubah.

Worker berjalan di proses lokal terpisah. Progres dan hasil fold disimpan di
direktori state aplikasi (`GTEX_DESKTOP_STATE_DIR` dapat menggantinya). Tombol
**Pause** menangguhkan worker beserta fit yang masih berjalan di memori. **Resume**
meneruskan proses yang sama tanpa mengulang fit. Jeda berlaku selama worker masih
hidup, bukan setelah restart komputer. Waktu jeda tidak dihitung dalam waktu proses
aktif. **Stop** mengakhiri worker, termasuk worker yang sedang dijeda. Fold yang selesai
tetap tersimpan, sedangkan fit yang terputus diulang saat proses dilanjutkan
dengan **Continue from checkpoint**, konfigurasi dan identitas data yang sama. Progres putaran menunjukkan
fit XGBoost saat ini, terpisah dari target putaran per fit.

Model berhasil muncul pada pilihan **Model prediksi** di Prediksi Jaringan.
Pergantian model mewajibkan audit input ulang. Tombol **Gunakan contoh lokal**
memakai `data/demo_tpm_shared.csv` yang tetap untuk semua model: gabungan enam
sampel unik dan 5.000 gen dari contoh yang sudah tersedia. Saat model diganti,
input contoh diaudit ulang dengan model baru, sampel aktif dipertahankan dan
hasil prediksi lama dihapus. Pengukuran yang tidak tersedia tetap kosong dan
mengikuti aturan imputasi model. Contoh gabungan mencakup sampel demo asli dan
contoh dari data training, sehingga bukan dataset evaluasi independen.
CSV mencatat `model_id`, `run_id`,
dan `boosting_rounds`. Hasil training-only tidak diberi skor CV/holdout lama.
Panel Gen menyediakan panel final dan panel setiap fold yang selesai, beserta skor
seleksi yang benar-benar tersimpan. Riwayat lama yang tidak menyimpan skor fold
hanya menampilkan daftar gen, tanpa membuat ulang atau mengarang skor.

Ekspor `training_data/` memuat development saja dan tidak memerlukan GCT asli
untuk runtime opsi notebook. Training dari unggahan tetap dapat digunakan bila
ekspor ini tidak tersedia. Gunakan `python desktop.py --check-training`
untuk memeriksa worker dengan fit ANOVA singkat, bukan mengubah model asli.

## Struktur

```text
main.py                   Kerangka aplikasi: pengaturan halaman, workspace, aplikasi utama
desktop.py                Launcher desktop
cloud_app.py              Launcher hosted
gtex_explorer/
  ui/
    app.py                Aplikasi utama setelah workspace dibuka
    workspace.py          Pemilihan dan pengelolaan workspace
    pages/                Satu berkas untuk setiap halaman
    shell/                Tema, navigasi, dan data bersama antarhalaman
    styles/               CSS aplikasi
  inference/              Parsing, audit input, dan prediksi jaringan
  experiments/            Tabel hasil, panel gen, dan diagnostik notebook
  training/               Data, konfigurasi, worker, checkpoint, gallery
  platform/
    cloud/                Penyimpanan privat, unggahan, daemon, ekspor
    paths.py              Lokasi resource aplikasi
    desktop_runtime.py    Server dan jendela desktop
    model_registry.py     Publikasi dan pemilihan model
    workspaces.py         Penyimpanan workspace
scripts/                  Ekspor notebook dan persiapan deployment
tests/                    Pengujian otomatis
resources/                Data notebook terkompresi untuk distribusi
data/                     Tabel hasil eksperimen dan contoh input
model_bundle/             Model terpilih
baseline_bundle/          Model baseline
```

Instalasi baru dapat memulihkan data development notebook tanpa GCT asli:

```powershell
python -m scripts.bundled_training_data restore
```

Resource disimpan dalam potongan terkompresi di bawah batas ukuran GitHub.
Pemulihan memeriksa checksum potongan dan hasil akhir sebelum publikasi.
`training_data/` yang sudah tersedia tetap dapat digunakan. Dataset unggahan,
workspace, model pengguna, kredensial, environment, build, dan cache tidak
termasuk repository.

## Menyiapkan model

### Putaran boosting pada Prediksi Jaringan

Pilih 100, 200, atau 300 putaran dari model tersimpan, dengan default 300.
Probabilitas dan kontribusi gen menggunakan pilihan yang sama. Mengubah pilihan
menghapus hasil lama tanpa mengulang audit input. CSV mencatat `boosting_rounds`.
Tidak ada pelatihan ulang, perubahan gen model, atau perubahan hasil CV/holdout.

### Diagnostik pada Panel Gen

Panel Cross-validation menampilkan diagnostik dari data training metode/fold yang dipilih,
terpisah dari skor prediksi pada data validation. ANOVA menyediakan η², Brown–Forsythe,
rasio variansi, residual, dan pembanding Welch. MI menyediakan skor pada 3, 5, dan 10
tetangga beserta perubahan ranking. L1 menyediakan jumlah gen nonzero, sparsitas gen,
konvergensi, iterasi, dan cutoff. Ringkasan Jaccard memakai 10 pasangan unik dari lima
fold pada metode dan top-k yang sama. Panel final hanya memakai diagnostik fit final.

Panel Gen menggunakan tab **Daftar gen**, **Diagnostik metode**, dan, untuk
selector CV, **Stabilitas antar-fold**. Daftar gen menempati lebar halaman dengan
filter dan unduhan CSV di bagiannya sendiri. Diagnostik dan skor XGBoost pada
data validation tetap dibedakan. Mengubah pencarian gen tidak mengubah ringkasan
diagnostik seluruh panel atau Jaccard antar-fold.

Panel final tidak menampilkan skor validation atau Jaccard CV. Baseline memiliki
tab **Kinerja XGBoost**, tanpa diagnostik selector. Jumlah sampel tiap sisi fold
dibaca dari metadata partisi CV tersimpan, bukan disalin dari gambar desain.
Target visual ada pada `design/gene-tabs-target-20261020.png`.

Untuk menyiapkan ulang ekspor diagnostik setelah hasil notebook diganti, jalankan
`.venv/Scripts/python.exe -m scripts.export_selector_diagnostics` dari direktori aplikasi.
Ekspor memeriksa fingerprint, skor, ranking, hasil fit, dan pasangan subset sebelum
menyimpan berkas serta manifest hash. Aplikasi tidak membaca checkpoint joblib untuk
diagnostik dan tidak menghitung ulang estimasi MI atau melatih selector/XGBoost.

### Ekspor model

Jalankan dari environment tesis yang memiliki pandas, NumPy, joblib, dan
XGBoost 3.2.0:

```powershell
python -m scripts.export_model_bundle
```

Script harus selesai dengan `status: validated` dan perbedaan metrik maksimum
terhadap notebook tidak lebih dari `1e-10`.

Baseline `XGBoost — All 5,000 Genes (No Feature Selection)` hanya menjadi
pembanding eksperimen. Model inferensi aplikasi tetap konfigurasi seleksi fitur
terbaik yang dilatih ulang pada seluruh data development.

## Panel gen dari cross-validation

Halaman Panel Gen memisahkan panel model final dari panel data latih CV. Pilih
ANOVA, Mutual Information, atau L1 dengan C 1, 0,1, atau 0,01, lalu pilih jumlah
500, 1.000, atau 2.000 gen dan fold 1–5. Kelima fit baseline tersedia sebagai
daftar seluruh 5.000 kandidat gen tanpa peringkat seleksi. Metrik yang ditampilkan
pada tampilan CV merupakan hasil validasi fold tersebut, bukan hasil holdout.

Peringkat berasal dari selector, bukan feature importance XGBoost. Kelas dominan
hanya tersedia untuk L1, berdasarkan koefisien absolut terbesar lintas kelas.
Untuk konfigurasi L1 yang perlu mengisi top-k dengan gen berskor nol, aplikasi
menampilkan peringatan dan tidak memberikan kelas dominan pada gen tersebut.

Ekspor tersimpan pada `data/cv_gene_rankings.csv.gz` dan
`data/cv_gene_panel_manifest.json`. Untuk memperbaruinya dari checkpoint lokal:

```powershell
.\.venv\Scripts\python.exe -m scripts.export_cv_gene_panels
```

Ekspor memeriksa fingerprint eksperimen, identitas fold, kandidat gen, 75 subset
gen yang benar-benar digunakan pada fit, dan 5 fit baseline. Tidak ada model
yang dilatih ulang. Aplikasi membaca CSV, bukan memuat checkpoint besar.
Unduhan mengikuti filter yang sedang aktif dan memuat identitas konfigurasi/fold.

## Kamus Biologi

Menu Kamus Biologi hanya memuat biologi molekuler, 13 jaringan target, dan
pengukuran ekspresi. Indeks di kiri membuka satu definisi di panel kanan.
Pencarian menerima nama Indonesia, istilah Inggris, dan label jaringan GTEx.
Istilah terkait membuka definisi lain dan menghapus filter yang menghalangi tujuan.
Unduhan CSV memuat seluruh hasil pencarian, bukan hanya definisi yang sedang terbuka,
beserta alamat rujukannya. Definisi dan rujukan dapat dibaca tanpa internet.

Pada Perbandingan Metode, pesan L1 diperjelas menjadi “Gen berkoefisien nonzero
kurang dari jumlah top-k pada satu atau lebih fold” ketika diagnostik tersimpan
memang menunjukkan kekurangan tersebut. Ini perubahan keterangan tampilan, bukan
perubahan metrik, aturan eligibility, hasil fit, atau checkpoint notebook.

## Menjalankan aplikasi

Versi **desktop Windows offline** tersedia melalui
`dist/GTExTissueExplorer/GTExTissueExplorer.exe` setelah build. Versi ini membuka
jendela aplikasi sendiri dan menyertakan Python, model, serta aset lokal.
Salin seluruh folder portable, bukan hanya `.exe`. Lihat [DESKTOP.md](docs/DESKTOP.md)
untuk petunjuk, persyaratan runtime Windows, dan cara membangun ulang.
Untuk pengembangan, gunakan `run_desktop_app.bat` setelah memasang
`requirements-desktop.txt`.

Environment aplikasi berada di `.venv`, terpisah dari venv notebook. Klik dua kali
`run_offline_app.bat`, atau jalankan `./run_offline_app.ps1`. Launcher menguji
interpreter dan impor dependensi, bukan hanya keberadaan `python.exe`.

Untuk menyiapkan ulang di komputer lain (dengan Python yang terinstal), jalankan
dari folder aplikasi:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_offline_app.ps1
```

Aplikasi hanya mendengarkan koneksi lokal pada `127.0.0.1:8501`. Telemetri
Streamlit dinonaktifkan melalui `.streamlit/config.toml`.
Pemasangan paket pertama dapat memerlukan internet. Setelah environment tersedia,
launcher dan inferensi tidak mengunduh paket/data. Mode browser ini memerlukan
Python. Versi desktop portable di atas sudah menyertakan runtime Python.

Pemeriksaan launcher tanpa membuka server:

```powershell
.\run_offline_app.ps1 -CheckOnly
.\run_offline_app.bat -CheckOnly
```

Opsional: set `THESIS_PYTHON` ke path interpreter lain yang memiliki semua
dependensi. Jika interpreter itu gagal, launcher mencoba `.venv` aplikasi,
venv tesis, venv lama aplikasi, lalu Python pada PATH. Venv yang rusak dilewati.

Pengujian otomatis:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Format input

Aplikasi menerima CSV/TSV/GCT dalam salah satu orientasi:

1. baris = sampel dan kolom = Ensembl gene ID, dengan kolom pertama `SAMPID`
2. baris = gen dan kolom = sampel, dengan kolom pertama berisi Ensembl gene ID

Nilai harus berupa TPM mentah non-negatif. Suffix versi Ensembl dapat berbeda
karena pencocokan dilakukan menggunakan stable ID sebelum titik.

Input diperiksa sebelum prediksi. Nilai negatif, tak hingga, token non-numerik,
dan ID duplikat tetap ditolak. Kecocokan ID gen model hanya ditampilkan sebagai
informasi, termasuk jika tidak ada gen yang cocok. Gen di luar model tidak digunakan.
Nilai gen model yang tidak tersedia diisi dengan median training tersimpan.

Ringkasan audit membedakan kecocokan ID, kelengkapan nilai gen model, dan ekspresi
nonzero pada sampel yang dipilih. Nonzero berarti TPM > 0 dari seluruh gen input
per sampel. Kelengkapan menghitung nilai gen model yang tersedia sebelum imputasi,
termasuk TPM = 0. Kedua ambang pengingat dapat diatur 0–100%, default 90%, tanpa
memblokir prediksi. Pilihan ambang dan model tetap diingat saat berpindah halaman.
Seluruh nilai model yang diimputasi dinyatakan secara eksplisit. Matriks semua nol
atau berisi nilai kosong tetap dapat diproses, sedangkan berkas tanpa struktur data
tetap ditolak. Definisi lengkap ada di `docs/prediction-input-policy.md`.

Mengganti atau menghapus unggahan menghapus audit dan prediksi lama. Perubahan
dideteksi dari SHA256 isi berkas, termasuk jika nama dan ukuran tetap sama.
Audit berulang membaca ulang isi dari awal. Kelengkapan/imputasi per sampel dapat
dilihat pada bagian audit.

`data/final_test_summary.csv` disinkronkan dari ekspor Part A-only dan harus
memuat satu model XGBoost terpilih. Data/model cache diperbarui pada rerun ketika
file ekspor berubah. Hasil CV dan holdout tetap berasal dari notebook, tidak
dihitung ulang di halaman Ringkasan.

## Tampilan hasil prediksi

Ringkasan menampilkan kelas terpilih, kelas peringkat kedua, dan jumlah probabilitas
11 kelas lainnya. Grafik menampilkan seluruh 13 kelas pada skala 0–100 persen,
dengan label persentase di samping setiap baris. Selisih dua probabilitas tertinggi
dinyatakan dalam poin persentase. Tampilan mengikuti sampel yang dipilih tanpa
mengubah keluaran model. Referensi lokasi jaringan tidak ditampilkan.

## Kohort pada Ringkasan

Ringkasan membedakan seluruh kohort dari dua subset berbasis donor. Jumlah donor
unik seluruh kohort adalah penjumlahan donor development dan donor holdout,
karena keduanya tidak tumpang tindih. Validasi silang hanya menggunakan data
development. Setelah konfigurasi dipilih, model dilatih ulang pada seluruh
data development dan dievaluasi pada holdout.

Bagian “Asal angka dan rincian eksperimen” menjelaskan SAMPID, SUBJID, SMTSD,
konfigurasi, pasangan konfigurasi–fold, serta berkas asal metrik. Data tampilan
tetap dibaca dari ekspor lokal, bukan angka yang ditetapkan dalam antarmuka.

Ringkasan memakai tab **CV Folding** dan **Hasil akhir - holdout**, dengan tab
holdout aktif saat pertama dibuka. Skor CV adalah rerata Macro-F1 dan simpangan
baku konfigurasi terpilih pada lima fold, bukan skor holdout.
Grafik F1 per jaringan menampilkan seluruh kelas dan jumlah sampel uji (n),
dengan pilihan urutan terendah, tertinggi, atau alfabetis. Precision dan recall
tambahan tersedia pada “Metrik lengkap holdout”. Tombol di bagian bawah membuka
halaman perbandingan, panel gen, atau prediksi yang sudah ada.

Pada Perbandingan Metode, metrik pilihan mengubah grafik dan kolom nilai tabel.
Bagian pertama memuat CV, ringkasan empat konfigurasi pada jumlah gen terpilih,
dan tabel lengkap 16 konfigurasi yang dapat dibuka. Bagian kedua memuat hasil
holdout untuk konfigurasi terpilih dan baseline, dengan persentase langsung pada
grafik. Titik CV berongga menandai konfigurasi yang tidak memenuhi syarat seleksi.
Peringkat Macro-F1 dan konfigurasi final tetap mengikuti ekspor notebook, bukan
dipilih ulang di aplikasi. Grafik memakai skala 0–1 dengan opsi perbesaran yang
ditandai. Panel Gen menjelaskan rentang peringkat pencarian. Halaman Prediksi
memakai identitas selector dan jumlah gen dari bundle, tanpa footer identitas
model yang diisi manual.

Istilah antarmuka memakai **data development**, **data training**, dan **data
validation** secara konsisten. Kamus memuat definisi ketiga subset tersebut.
Target visual yang disetujui disimpan di `design/overview-tabs-target-20261020.png`
dan `design/comparison-cv-holdout-target-20261020.png`. Status verifikasi visual
dicatat terpisah dari pengujian fungsional di `docs/design-qa.md`.

Metrik evaluasi ditampilkan sebagai persentase dengan dua desimal, termasuk
label, tooltip, tabel, dan metrik validasi pada Panel Gen. Contoh Macro-F1
0,9625 ditampilkan sebagai 96,25%. Simpangan baku dan selisih dua skor memakai
poin persentase. Nilai numerik pada ekspor sumber dan model tetap skala 0–1.
Statistik F ANOVA, MI, koefisien L1, serta kontribusi gen tidak dikonversi ke persen.

Dasar pola antarmuka dan batas pemeriksaan tercatat pada
`design/experimental-ui-design-review.md`. Aplikasi tidak bergantung pada atau
menghubungi platform yang menjadi referensi desain.

### Penyimpanan hasil ke gallery

Alurnya adalah training, penyimpanan model beserta metrik dan panel gen,
verifikasi berkas, lalu publikasi ke gallery. Halaman Hasil pelatihan memiliki
pilihan **Gallery model** untuk hasil yang sudah terverifikasi dan **Status proses**
untuk riwayat proses, termasuk yang berjalan, Pause, Stop, atau gagal.

Pemeriksaan mencakup identitas konfigurasi dan dataset, hash berkas model,
kesesuaian panel final dengan gen model, serta kelengkapan metrik dan panel setiap
fold jika CV diaktifkan. Ringkasan CV dihitung ulang dari hasil fold tersimpan
untuk memeriksa konsistensinya, tanpa melatih ulang model. Training tanpa CV
disimpan tanpa skor validation atau holdout baru.

Indeks `runs/<run_id>/gallery.json` ditulis secara atomik setelah verifikasi.
Model pengguna baru muncul pada gallery, Panel gen pelatihan, dan pilihan model
prediksi setelah indeks tersedia dan status proses selesai. Selama penyimpanan
dan verifikasi akhir, Pause dan Stop dinonaktifkan agar pembaruan dapat selesai.
Hasil lama dengan status selesai diperiksa menggunakan aturan yang sama. Hasil
yang belum lengkap tetap tersedia dalam riwayat proses. Percobaan ulang memakai
model dan checkpoint yang sudah tersimpan jika identitas serta hash cocok.

Gallery menggunakan daftar model yang dapat dicari berdasarkan metode, parameter
selektor, jumlah gen, atau nama dataset. Pilihan model ditampilkan di panel kanan,
dengan tombol prediksi dan Panel Gen sebelum metrik CV. Tab Status proses
menampung pemantauan dan kontrol proses. Pada jendela yang lebih sempit, daftar
dan rincian disusun vertikal. Seluruh isi tetap berasal dari hasil tersimpan.

## Deployment Vercel

Versi hosted memakai `cloud_app.py`, container web, layanan unggahan privat,
serta worker terpisah (`python -m gtex_explorer.platform.cloud.daemon`). Docker memulihkan
data notebook dari `resources/` saat build. Jangan menyimpan `.env.local` atau
`.vercel/` ke Git. Koneksi Git memakai repository privat dan perlindungan
Vercel Authentication tetap aktif.

Status terakhir dan batas operasional tersedia dalam
`docs/deployment-readiness-vercel.md`. Build berhasil tidak berarti penyimpanan
dan training cloud telah diverifikasi. Worker harus tetap nonaktif selama
backing store ditangguhkan.
