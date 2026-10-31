# Rencana Pengujian

## Tambahan pengujian pelatihan lokal

- Identitas data, ekspor development-only dan pencegahan kebocoran validation.
- Kontrol ANOVA/MI/L1/baseline, dukungan nonzero L1 dan konvergensi.
- Worker tunggal, pembatalan, resume, pemulihan worker mati, serta penulisan atomik.
- Model tersimpan, integritas bundle, contoh model dan round-trip prediksi.
- Pergantian model menghapus audit/hasil/putaran lama.
- UI selector, CV default nonaktif, cakupan kelas untuk pilihan fold.
- `desktop.py --check-training` menjalankan worker nyata tanpa mengganti model asli.
- Build baru diperiksa dengan `--check` dan `--check-training` dalam mode offline.

## 1. Pengujian unit

- penghapusan suffix versi Ensembl
- parsing CSV dengan orientasi sampel × gen
- parsing TSV dengan orientasi gen × sampel
- penolakan SAMPID duplikat
- pencocokan stable ID antar-versi
- imputasi nilai kosong dengan median development
- penolakan nilai tak hingga
- penolakan stable ID gen duplikat

Perintah: `python -m pytest -q`.

## 2. Pengujian integrasi model

1. Muat `preprocessor.joblib` dan `xgboost_model.json`.
2. Pastikan 1.000 gen, 13 kelas, serta panjang mean, scale, dan median konsisten.
3. Cocokkan SHA256 model dengan `bundle_manifest.json`.
4. Prediksi `data/demo_tpm.csv`.
5. Pastikan setiap probabilitas finite, berada pada rentang 0 sampai 1, dan jumlah probabilitas per sampel mendekati 1.
6. Pastikan kelas prediksi sama dengan argmax probabilitas.

## 3. Pengujian integritas eksperimen CV

1. Pastikan 15 konfigurasi selector tetap menghasilkan 75 baris.
2. Pastikan baseline `xgboost_all_5000` menghasilkan lima baris dan memakai 5.000 gen pada setiap fold.
3. Pastikan gabungan memuat 16 konfigurasi × 5 fold = 80 hasil.
4. Pastikan setiap pasangan `configuration_id` dan `fold` unik dan lengkap.
5. Cocokkan fingerprint fold dan parameter XGBoost untuk seluruh konfigurasi.
6. Pastikan tidak ada donor yang muncul pada train dan validation dalam fold yang sama.
7. Hitung ulang metrik baseline dari prediksi per sampel dan cocokkan agregasinya.

## 4. Pengujian alur antarmuka

- aplikasi terbuka tanpa exception atau error console
- empat item navigasi membuka halaman yang tepat
- data contoh lokal dapat dimuat dan diaudit
- tombol prediksi tidak aktif sebelum audit lolos
- pemilihan SAMPID mengubah hasil, grafik, dan tabel kontribusi
- unggah data demo menghasilkan status audit berhasil
- unduh hasil menghasilkan CSV berisi label dan 13 probabilitas
- pencarian dan filter panel gen berfungsi
- pemilihan metrik mengubah grafik perbandingan
- garis baseline 5.000 gen, ringkasan metrik, dan tabel 16 konfigurasi tampil pada halaman Perbandingan Metode

## 5. Pengujian kondisi gagal

- file kosong
- format dengan hanya satu kolom
- TPM negatif
- nilai tak hingga
- stable ID gen duplikat setelah suffix versi dihapus
- SAMPID duplikat
- token non-numerik pada CSV, TSV, dan GCT

Kondisi penolakan di atas harus menampilkan pesan yang dapat ditindaklanjuti
dan tidak menjalankan prediksi.

## Pengingat input dan perubahan layout

Kasus berikut memeriksa pengingat, bukan penolakan prediksi:

- kecocokan ID gen di bawah 90% atau nol gen cocok
- matriks semua nol, nilai model kosong, dan kombinasi kolom hilang/nilai kosong
- perbedaan persentase gen model cocok, kelengkapan model, dan TPM > 0 dari seluruh gen input
- ambang nonzero/kelengkapan 0%, 50%, 90%, dan 100% tanpa mengubah input model atau prediksi
- pilihan ambang sebelum audit pertama dan setelah berpindah halaman
- pilihan model selain default tetap dipertahankan saat navigasi
- angka ringkasan berubah mengikuti sampel terpilih tanpa mengganti kecocokan ID file
- tabel ringkas menampilkan nilai pada lebar desktop normal tanpa scroll horizontal

Pengujian black-box formal tetap dijeda. Catatan regresi pengembangan layout terbaru
terdapat pada `design/prediction_layout_20261024/`, terpisah dari bukti penerimaan formal.

### Pemeriksaan regresi tambahan

- median pecahan pada kolom nullable integer
- penggantian berkas dengan nama/ukuran sama tetapi isi berbeda
- penghapusan unggahan dan perpindahan dari demo ke unggahan baru
- audit berulang pada unggahan yang sama
- pencarian gen dengan karakter regex seperti `[`, `(`, dan `*`
- tabel Ringkasan tanpa DummyClassifier dan sesuai ekspor Part A-only
- legend selector tampil untuk kelima pilihan metrik
- launcher PowerShell/BAT berhasil memeriksa runtime dengan `-CheckOnly`

## 6. Design QA

### Putaran boosting pada Prediksi Jaringan

- Default 300 putaran identik dengan probabilitas model penuh sebelum perubahan.
- Pilihan 100/200/300 memberi probabilitas dan kontribusi gen sesuai `iteration_range` yang sama pada XGBoost native.
- Pilihan di luar rentang, pecahan, boolean, dan string ditolak oleh API inferensi.
- Perubahan pilihan mempertahankan audit input tetapi menghapus prediksi, grafik, kontribusi, dan unduhan sebelumnya.
- Setelah prediksi dijalankan kembali, CSV memuat jumlah putaran dan probabilitas yang sama dengan hasil pada halaman.
- Perpindahan halaman tidak boleh mencampur hasil tersimpan dengan pilihan putaran lain.
- Periksa susunan selector/run/download dan penanda putaran pada hasil. Hasil CV dan holdout tidak berubah.

### Pemisahan CV dan holdout

- Ringkasan membuka tab holdout secara default. Skor CV dan grafik holdout berada dalam tab berbeda.
- Mengubah metrik holdout pada fixture tidak mengubah skor CV, dan angka tetap berasal dari ekspor.
- Perbandingan Metode memisahkan dua tahap, menyimpan semua 16 konfigurasi dan alasan kelayakannya dalam detail CV.
- Kontrol metrik mengubah grafik dan ringkasan CV, bukan konfigurasi model final atau hasil holdout.
- Grafik holdout menampilkan dua model, lima metrik, persentase langsung, dan warna model yang konsisten.
- Cocokkan tampilan normal/narrow dengan dua target visual 20 Oktober 2026. AppTest bukan bukti kesesuaian visual.
- Teks yang dihapus pada referensi tidak muncul kembali. Gunakan data development, data training, dan data validation.

### Diagnostik Panel Gen

- Tabel, filter, dan unduhan CSV berada di Daftar gen, bukan di antara metrik diagnostik.
- Pisahkan diagnostik metode dan skor validation dari ringkasan Jaccard lintas seluruh fold.
- Panel final tidak menampilkan skor validation atau tab stabilitas CV. Baseline tidak mendapat diagnostik selector.
- Perubahan fold memperbarui jumlah sampel training/validation serta hasil fit, tanpa mengubah Jaccard seluruh fold.
- Unduhan panel gen mengikuti tabel dan mencatat metode, top-k, fold, serta sumber panel, terpisah dari unduhan diagnostik.
- Periksa ANOVA, MI, dan tiga nilai C L1 pada beberapa fold dan ukuran top-k.
- Ringkasan metode tidak mengikuti pencarian, sedangkan tabel diagnostik gen dan CSV mengikuti pencarian/peringkat tampilan.
- ANOVA membedakan median η² dan flag variansi pada top-k dari Spearman seluruh kandidat. MI membedakan cutoff top-k dari jumlah gen positif dan Spearman seluruh kandidat.
- Nonzero, sparsitas, konvergensi, dan iterasi L1 menjelaskan fit seluruh kandidat sebelum pemilihan top-k. Tampilkan jumlah nonzero/nol dalam panel top-k secara terpisah, termasuk pada model final.
- Cocokkan η², flag Brown–Forsythe, dan variansi dengan ringkasan dan tabel gen notebook.
- Bedakan sensitivitas MI antar-neighbors pada fold yang sama dari Jaccard antar-fold.
- Pastikan sparsitas gen = 1 − proporsi gen nonzero, bukan proporsi nonzero itu sendiri.
- Panel final tidak mengambil iterasi/nonzero dari fold CV. Baseline tidak mempunyai diagnostik selektor.
- Rerata/minimum Jaccard dihitung dari 10 pasangan unik, tanpa diagonal. Mengubah fold aktif tidak mengubah ringkasan lima fold.
- Pertahankan status konfigurasi L1 yang top-k-nya mencakup koefisien nol.
- Tolak ekspor dengan fingerprint, hash, pasangan fold, atau hasil fit yang tidak sesuai.
- Jika seluruh ekspor diagnostik belum tersedia, tetap tampilkan panel gen dan pesan ketersediaan, bukan nilai nol buatan.
- Pemeriksaan otomatis bukan penerimaan visual. Tinjau kolom gen dan metrik pada lebar normal dan sempit sebelum penerimaan tata letak.

Bandingkan implementasi pada viewport 1440 × 1024 dengan visual audit-gate option 2. Pemeriksaan wajib meliputi tipografi, ritme layout, warna, kualitas aset, copy, state kontrol, dan keterbacaan grafik. Temuan P0 sampai P2 harus diperbaiki sebelum handoff.

### Kamus Biologi

- Cocokkan indeks dan panel definisi dengan `design/biology-dictionary-target-20261020.png`.
- Seluruh 13 label jaringan GTEx harus dapat dicari dengan nama Indonesia dan Inggris.
- Batasi isi pada biologi molekuler, jaringan/anatomi, serta pengukuran ekspresi.
- Periksa perubahan istilah, filter kategori, pencarian kosong/tidak cocok, dan tautan istilah terkait.
- Jika istilah terkait berada di luar filter aktif, buka istilah tersebut dan reset filter.
- Unduhan memuat seluruh hasil filter beserta judul dan alamat rujukan, bukan hanya definisi terbuka.
- Pastikan perubahan ini tidak mengubah model, metrik eksperimen, atau pilihan putaran prediksi.
