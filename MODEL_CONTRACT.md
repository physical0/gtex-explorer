# Kontrak Model dan Data

## Identitas model

- Pipeline: Part A GTEx V11
- Selektor final: L1-Regularization dengan C = 1
- Fitur final: 1.000 gen
- Klasifikator: XGBoost multikelas
- Kelas: 13 jaringan kardiometabolik dan endokrin
- Data development: 807 sampel dari 494 donor
- Data uji akhir: 193 sampel dari 124 donor yang tidak pernah masuk development
- Fingerprint eksperimen: `827073e65504bf4fc82a621aa3bee47177f1f2552aaf705cfc90ff847e7b7729`

## Kontrak input

Input harus berupa matriks TPM mentah non-negatif. Format yang didukung:

1. Sampel × gen, dengan kolom pertama berisi SAMPID.
2. Gen × sampel, dengan kolom pertama berisi Ensembl gene ID.
3. GCT 1.2 kecil dengan dua baris header.

Nama gen dicocokkan setelah suffix versi Ensembl dihapus. Sebagai contoh, `ENSG00000170549.4` cocok dengan `ENSG00000170549.9`.

Aturan audit:

- minimal 900 dari 1.000 gen model harus cocok
- TPM negatif ditolak
- nilai positif atau negatif tak hingga ditolak
- stable ID gen duplikat ditolak
- nilai kosong pada kolom gen yang cocok diganti median TPM development
- gen model yang tidak tersedia diganti median TPM development, selama ambang kecocokan tetap terpenuhi
- token ekspresi non-numerik ditolak, bukan diam-diam diubah menjadi nilai kosong
- setiap sampel wajib memiliki setidaknya 90 persen nilai gen model terukur sebelum imputasi
- nilai kosong dan kolom tidak tersedia digabung dalam batas maksimum 10 persen imputasi per sampel
- sampel tanpa nilai TPM positif pada gen model (semua nol/kosong) ditolak
- jika satu sampel gagal, prediksi seluruh unggahan diblokir sampai input diperbaiki

Ambang 90 persen adalah kebijakan kelengkapan input, bukan ambang biologis yang
dipelajari dari GTEx. Lolos QC tidak menjamin sampel berada dalam distribusi model.

## Preprocessing beku

Urutan inferensi tidak boleh diubah:

1. Selaraskan kolom ke urutan tepat 1.000 gen model.
2. Imputasi nilai yang tidak tersedia dengan median TPM development.
3. Terapkan `log2(TPM + 1)`.
4. Terapkan standardisasi menggunakan mean dan standard deviation data development yang tersimpan.
5. Kirim matriks hasil ke XGBoost.

Tidak ada PCA dan tidak ada seleksi fitur ulang saat inferensi.

## Putaran boosting saat inferensi

Halaman Prediksi menyediakan pilihan 100, 200, atau 300 putaran awal dari model
tersimpan. Nilai default 300 memakai model penuh dan mempertahankan prediksi
sebelumnya. Pilihan ini diterapkan melalui `iteration_range=(0, boosting_rounds)`
pada probabilitas dan kontribusi lokal gen, tanpa mengubah bobot model atau
melakukan pelatihan ulang. Putaran di luar rentang model ditolak.

Perubahan pilihan menghapus hasil dan unduhan lama, tetapi mempertahankan audit
input. Kolom `boosting_rounds` pada CSV mencatat jumlah putaran yang menghasilkan
prediksi tersebut. Hasil CV dan holdout pada halaman lain tetap berasal dari
eksperimen tersimpan dengan model penuh, bukan dihitung ulang dari pilihan ini.

## Kontrak output

Untuk setiap SAMPID, aplikasi menghasilkan:

- `predicted_tissue`
- `top_probability`, yaitu probabilitas kelas tertinggi
- `boosting_rounds`, yaitu jumlah putaran awal model yang dipakai
- probabilitas untuk seluruh 13 kelas jaringan
- kontribusi lokal fitur XGBoost untuk sampel yang sedang ditinjau

Probabilitas adalah keluaran model untuk perbandingan relatif antar-kelas. Nilai ini bukan probabilitas klinis dan belum dikalibrasi untuk penggunaan klinis.

Kontribusi lokal fitur berada pada skala margin model. Nilai tersebut menjelaskan
fitur yang paling memengaruhi keluaran XGBoost untuk satu sampel dan bukan bukti
hubungan sebab-akibat biologis.

## Bukti reproduksibilitas

Halaman Perbandingan Metode juga membaca baseline
`XGBoost — All 5,000 Genes (No Feature Selection)`. Baseline memakai matriks
praproses 5.000 gen tanpa pemeringkatan atau pemotongan top-k dan menggunakan
parameter serta lima split donor yang sama. Baseline hanya menjadi referensi CV
dan bukan model inferensi yang diekspor.

Manifest bundle mencatat Macro-F1 0,962502, balanced accuracy 0,954895, dan accuracy 0,963731 pada data uji donor-disjoint. Hash SHA256 model adalah `e9fd736a9bcdaa90ba715c5255c8c29786f6ca84a200d75f3734c6db13a8fe77`.
