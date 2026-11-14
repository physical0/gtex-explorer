# Kebutuhan Aplikasi HTx Tissue Classifier

## Tujuan

Tambahan pelatihan lokal: memilih selector/top-k, parameter selector dan XGBoost,
CV donor-grouped opsional, penyimpanan model baru, pembatalan/resume, serta
pemilihan bundle pada Prediksi Jaringan. Hasil eksperimen asli tetap terpisah.
Training memakai development saja. Holdout tidak digunakan untuk tuning.

Aplikasi merupakan alat penelitian dan demonstrasi Part A untuk:

- menampilkan ringkasan eksperimen seleksi fitur
- membandingkan ANOVA, Mutual Information, dan L1-Regularization dengan baseline XGBoost seluruh 5.000 gen
- menjelajahi panel konfigurasi final serta panel metode, jumlah gen, dan fold dari seluruh 80 fit CV
- memprediksi satu atau beberapa kelas jaringan dari matriks TPM dengan model XGBoost final

Aplikasi tidak ditujukan untuk diagnosis, penentuan terapi, atau pengambilan keputusan klinis.

## Pengguna sasaran

Pengguna utama adalah mahasiswa peneliti, dosen pembimbing, dan penguji yang ingin meninjau hasil Part A atau mendemonstrasikan inferensi model tanpa menjalankan ulang notebook selama lebih dari dua jam.

## Ruang lingkup MVP

1. Halaman Ringkasan menampilkan jumlah sampel, donor, kelas jaringan, alur eksperimen, konfigurasi final, dan metrik uji akhir beserta selisihnya dengan baseline.
2. Tab Cross-validation pada Ringkasan membandingkan 15 konfigurasi selektor dan satu baseline tanpa seleksi fitur beserta variasi antar-fold. Halaman Perbandingan Metode yang terpisah digabungkan ke Ringkasan.
3. Halaman Panel Gen menyediakan panel final dan panel 75 fit selektor beserta lima fit baseline, dengan pencarian, filter metode/jumlah gen/fold, tabel, dan unduhan.
4. Halaman Prediksi Jaringan memisahkan audit matriks dari eksekusi model. Prediksi hanya dapat dijalankan setelah input lolos pemeriksaan.
5. Hasil prediksi menampilkan seluruh probabilitas kelas, selisih top-1 dan top-2, serta kontribusi lokal gen terhadap kelas prediksi.
6. Grafik hasil memuat seluruh 13 kelas, label persentase, dan skala tetap 0–100 persen. Referensi lokasi jaringan tidak ditampilkan.
7. Aplikasi berjalan lokal tanpa API, basis data, atau aset yang dimuat dari internet.
8. Halaman Kamus Biologi menyediakan indeks dan panel definisi, pencarian nama Indonesia/Inggris/GTEx, kategori, istilah terkait, rujukan, dan unduhan CSV yang dapat dipakai offline.

## Kriteria penerimaan

- Bundle model dapat dimuat dan hash model cocok dengan manifest.
- Data contoh lokal dapat diaudit dan diprediksi tanpa koneksi internet.
- Model tidak dapat dijalankan sebelum audit input lolos.
- Model menerima orientasi sampel × gen dan gen × sampel.
- Versi Ensembl dapat berbeda karena pencocokan menggunakan stable ID sebelum titik.
- Input ditolak jika nilai negatif, nilai tak hingga, atau stable ID gen duplikat ditemukan.
- Nilai kosong pada gen yang cocok diimputasi dengan median TPM data development.
- Kecocokan ID gen model bersifat informatif, termasuk nol gen yang cocok. Gen input di luar model tidak digunakan oleh model tersimpan.
- Ringkasan memisahkan kecocokan ID, kelengkapan nilai model sebelum imputasi, dan TPM positif dari seluruh gen input per sampel. TPM = 0 tersedia tetapi bukan positif.
- Ambang pengingat nonzero dan kelengkapan dapat diatur 0–100%, default 90%, tanpa memblokir prediksi. Pilihan ambang dan model dipertahankan saat navigasi.
- Matriks semua nol atau nilai model yang kosong dapat diproses dengan aturan imputasi tersimpan. Berkas tanpa struktur data dan token ekspresi non-numerik tetap ditolak.
- Mengganti/menghapus unggahan membatalkan audit dan prediksi lama, termasuk perubahan isi dengan nama/ukuran sama.
- Ringkasan memakai ekspor Part A-only tanpa baris DummyClassifier.
- Legend metode terlihat dan pencarian panel gen memakai substring literal.
- Halaman eksperimen dan training dikelompokkan terpisah, sedangkan Prediksi Jaringan dan Kamus Biologi tetap dapat diakses bersama. Unggah, pemilihan sampel, prediksi dan unduh hasil tetap berfungsi.
- Peringatan validasi input tetap ditampilkan. Teks peringatan klinis dan sebab-akibat pada hasil prediksi serta footer umum dihapus dari antarmuka.
- Metrik bundle mereproduksi hasil notebook dengan selisih maksimum 1e-10.
- Tabel eksperimen memuat 16 konfigurasi, lima fold per konfigurasi, dan tepat 80 hasil fold-level tanpa pasangan konfigurasi-fold yang hilang atau duplikat.
- Baseline muncul satu kali per fold, memakai semua 5.000 gen, dan tidak masuk pemeringkatan kandidat selector.
- Panel CV cocok dengan subset gen pada checkpoint fit dan tidak dipresentasikan sebagai panel model final.
- Gen berskor nol pada top-k L1 diberi peringatan. Kelas dominan tidak diisi pada gen yang tidak memiliki koefisien nonzero.
- ANOVA dan MI tidak diberi kelas dominan berbasis koefisien L1. Baseline tidak diberi skor atau peringkat seleksi.
- Keterangan kurangnya gen nonzero digunakan hanya untuk L1 ketika didukung diagnostik fold. Konvergensi atau penyebab lain tidak diganti dengan alasan yang tidak terbukti.
- Kamus hanya memuat konsep biologi molekuler, 13 jaringan target, dan pengukuran ekspresi. Istilah algoritma, metrik evaluasi, dan perangkat lunak tidak dimasukkan.
- Ringkasan membedakan kohort, split donor, skor CV konfigurasi terpilih, dan evaluasi holdout. Grafik jaringan memuat F1 serta jumlah sampel uji per kelas, dengan pengurutan yang tidak mengubah hasil.
- Konfigurasi final pada Ringkasan mengikuti selector dan jumlah gen yang tersimpan pada notebook, bukan dipilih ulang di aplikasi. Pilihan metrik memperbarui grafik dan kolom skor tabel.
- Grafik perbandingan memakai data skor 0–1 dengan label 0–100% secara default. Perbesaran rentang dinyatakan pada sumbu. Skor per fold, rerata, simpangan baku, dan probabilitas sampel tidak dipertukarkan.
- Metrik evaluasi ditampilkan dalam persen dengan dua desimal. Simpangan baku dan selisih skor dinyatakan dalam poin persentase. Skor seleksi gen dan berkas sumber tidak dikonversi.
- Identitas model pada halaman Prediksi berasal dari bundle. Tidak ada footer identitas model yang diisi manual atau referensi lokasi jaringan.

## Di luar ruang lingkup MVP

- pelatihan ulang model di dalam aplikasi
- integrasi database atau akun pengguna
- anatomi 3D mesh interaktif penuh
- koneksi API, sinkronisasi cloud, atau penyimpanan unggahan di server eksternal
- interpretasi biologis atau klinis individual
- dukungan microarray, raw counts, FPKM, atau data yang sudah ditransformasi log
