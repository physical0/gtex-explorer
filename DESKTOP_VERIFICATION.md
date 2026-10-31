# Verifikasi desktop offline

Tanggal: 10 Oktober 2026. Target pengujian: Windows 64-bit pada komputer pengembangan.

- 55 tes regresi lulus, termasuk pemeriksaan model, konsistensi hasil penelitian,
  halaman prediksi, pembatasan jaringan backend, dan lifecycle launcher.
- Pemeriksaan executable final (`--check`) lulus. Bundle memuat model 13 kelas
  dengan 1.000 gen, audit dan inferensi tiga sampel contoh berhasil, serta server
  dan halaman localhost tersedia. Proses server uji dihentikan setelah pemeriksaan.
- Pada pengujian jendela dari source, data contoh lolos audit dan prediksi muncul
  bersama grafik probabilitas serta kontribusi lokal gen. Dialog ekspor CSV terbuka.
- Pengujian antarmuka executable dihentikan melalui Escape. Setelah perbaikan
  pengaturan startup, executable diperiksa otomatis tanpa membuka jendela kembali.
  Uji visual final dan penyimpanan CSV melalui dialog masih perlu dikonfirmasi.

Pengujian tidak mematikan jaringan Windows. Koneksi eksternal backend diblokir
oleh launcher dan sumber frontend dibatasi melalui Content Security Policy.
Distribusi pada komputer Windows lain belum diuji. WebView2 Runtime dan .NET
Framework merupakan prasyarat sistem, bukan bagian dari paket portable.

Notebook, preprocessing, model XGBoost, dan hasil penelitian tidak diubah oleh
implementasi desktop ini.
