# Review isi dan rancangan antarmuka eksperimen

Tanggal: 17 Oktober 2026. Ruang lingkup: Ringkasan, Perbandingan Metode,
Panel Gen, Prediksi Jaringan, dan Kamus Istilah.

## Rujukan aplikasi

Rujukan berikut dibaca dari dokumentasi pengembang, bukan hasil pengujian
usability terhadap aplikasi ini. Pola diadaptasi untuk aplikasi desktop yang
sudah ada, tanpa memasang platform lain atau mengubah pipeline model.

| Rujukan resmi | Pola yang dijelaskan sumber | Penerapan lokal |
| --- | --- | --- |
| [MLflow Tracking](https://mlflow.org/docs/latest/tracking/) | Membedakan parameter, metrik, dataset, dan artefak suatu run. | Konfigurasi terpilih ditautkan ke ringkasan CV yang sama, metrik holdout dipisahkan, dan sumber berkas dapat ditinjau. |
| [GenePattern User Guide](https://www.genepattern.org/user-guide/) | Hasil suatu job menampilkan parameter input dan berkas hasil, serta dapat diteruskan ke analisis berikutnya. | Urutan data, pemilihan, hasil final, dan tindakan lanjutan dijelaskan secara ringkas. Tombol membuka halaman aplikasi yang sudah tersedia. |
| [UCSC Xena Visual Spreadsheet](https://ucsc-xena.gitbook.io/project/overview-of-features/visual-spreadsheet) | Pemilihan kohort/dataset dan konteks sampel menyertai eksplorasi data genomik, dengan unit dan informasi dataset yang dapat ditelusuri. | Konteks kohort dan split tetap dekat hasil. Grafik jaringan memperlihatkan F1 dan jumlah sampel uji, bukan hanya peringkat skor. |

Ini adalah interpretasi desain atas pola dokumentasi tersebut, bukan salinan
visual atau klaim bahwa tata letak lokal sudah terbukti meningkatkan usability.

## Temuan dari kode dan isi

1. Ringkasan sebelumnya mengulang penjelasan split, holdout, dan metode di beberapa
   tempat. Skor validasi konfigurasi terpilih belum ditampilkan dekat identitasnya.
2. Grafik jaringan menyimpan jumlah sampel hanya dalam tooltip. Grafik awal
   menonjolkan skor tertinggi, sedangkan jaringan dengan skor lebih rendah berada
   di bagian akhir.
3. Perbandingan Metode memilih ulang baris terbaik dari skor CV, alih-alih sekadar
   menampilkan pilihan yang sudah dikunci notebook. Kolom tabel tetap Macro-F1
   meskipun pengguna mengganti metrik pada grafik.
4. Panel Gen menerapkan pencarian setelah pembatasan peringkat. Cakupan tersebut
   perlu terlihat agar hasil kosong tidak disalahartikan sebagai gen yang tidak
   tersedia pada seluruh panel.
5. Prediksi Jaringan memiliki identitas model berulang, termasuk footer dengan
   selector dan jumlah gen yang diisi manual. Judul halaman belum memakai h1
   Streamlit yang konsisten dengan halaman lain.

Temuan ini adalah pemeriksaan kode/isi. Tanpa tangkapan layar baru, tidak dibuat
klaim temuan visual tentang pemotongan teks, posisi aktual, atau aksesibilitas penuh.

## Perubahan

- Ringkasan memakai urutan kohort dan split, pemilihan konfigurasi, model/CV,
  holdout, lalu hasil jaringan. Penjelasan metrik dan sumber dipisahkan dalam
  bagian yang dapat dibuka. Tiga metrik final utama tetap tersedia, sedangkan
  macro-precision dan macro-recall tidak diulang sebagai kartu tambahan.
- Baris CV model final dicari dengan selector **dan** jumlah gen. Rerata ± SD
  dipisahkan dari skor holdout. Tidak ada pemilihan menggunakan holdout.
- Grafik seluruh 13 jaringan memakai skala tetap 0–1, nama lengkap, nilai F1,
  jumlah sampel uji, dan kontrol pengurutan. Nilai bukan probabilitas sampel.
- Perbandingan menggunakan pilihan notebook. Grafik/tabel mengikuti metrik
  aktif, sementara peringkat dinyatakan tetap berdasarkan Macro-F1. Perbesaran
  rentang grafik bersifat opsional dan dinyatakan pada sumbu.
- Panel Gen mempertahankan kontrol metode/C/jumlah gen/fold, serta menjelaskan
  batas pencarian. Nilai dan subset gen tetap berasal dari ekspor fit.
- Prediksi menggunakan identitas bundle yang dinamis, judul halaman konsisten,
  SAMPID dekat hasil, dan penjelasan probabilitas serta kontribusi dalam unitnya
  masing-masing. Footer identitas model yang berulang dihapus.
- Kamus mempertahankan definisi, pencarian, kategori, pagination, dan ekspor.
  Intro dipersingkat dan keterangan label operasional hanya tampil pada kategori
  yang relevan.

## Batas dan pemeriksaan

Model, preprocessing, checkpoint notebook, ekspor numerik, dan definisi metrik
tidak diubah. Tidak ada ketergantungan daring baru atau pelatihan ulang.

Review visual formal belum dilakukan karena browser menolak observasi preview
lokal pada percobaan sebelumnya. Tidak digunakan browser alternatif, tangkapan
layar OS, atau jalur lain untuk melewati pembatasan tersebut. Uji otomatis
memeriksa rendering logis dan isi komponen, bukan penerimaan visual.

Sebanyak 115 uji regresi lulus, diikuti 18 uji terfokus pada snapshot akhir.
Review independen satu agen tidak menemukan masalah konkret pada pemetaan data
dan isi. Build desktop pada `dist/research_design_update/GTExTissueExplorer/`
lulus pemeriksaan model, prediksi contoh, dan startup server lokal. Hash lima
berkas halaman dalam paket sesuai dengan sumber yang direvisi.
