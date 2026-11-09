# Prediksi Hasil Laga Sepak Bola Internasional dengan Random Forest

Tugas data mining: memprediksi hasil laga sepak bola internasional dalam tiga kelas (menang kandang, seri, menang tandang) memakai Random Forest Classifier. Seluruh proses ada di [`Modeling.ipynb`](Modeling.ipynb).

## Data

- Sumber: `results.csv` dari kumpulan data [martj42/international_results](https://github.com/martj42/international_results), dibaca langsung dari URL.
- 48.943 laga mentah, disaring menjadi 15.362 laga sejak 2010.
- Kolom asli: tanggal, tim kandang, tim tandang, skor kandang, skor tandang, turnamen, kota, negara, dan status lapangan netral.

## Metode

1. Label `match_result` dibentuk dari skor: 2 = menang kandang, 1 = seri, 0 = menang tandang.
2. Fitur: kode tim kandang, kode tim tandang, dan kode turnamen hasil `LabelEncoder`.
3. Data dibagi 80:20 (`test_size=0.2`, `random_state=42`): 12.289 laga latih dan 3.073 laga uji.
4. Model: `RandomForestClassifier(n_estimators=100, random_state=42)`.
5. Evaluasi: akurasi, classification report per kelas, dan confusion matrix.

## Hasil pada data uji (n = 3.073)

| Kelas | Precision | Recall | F1 | Jumlah laga |
|---|---:|---:|---:|---:|
| Menang tandang | 0,39 | 0,39 | 0,39 | 853 |
| Seri | 0,23 | 0,17 | 0,20 | 716 |
| Menang kandang | 0,57 | 0,63 | 0,60 | 1.504 |

Akurasi: **45,88%**.

## Evaluasi terhadap baseline

Tebakan paling sederhana, yaitu selalu memprediksi menang kandang, benar pada 1.504 dari 3.073 laga uji (48,9%). Model ini masih sekitar 3 poin persentase di bawah baseline tersebut. Penyebab yang paling mungkin: kode tim hasil label encoding hanya nomor urut abjad dan tidak membawa informasi kekuatan tim. Kelas seri paling sulit ditebak (F1 0,20).

Rencana perbaikan:

- Fitur kekuatan dan performa tim, misalnya rating Elo, rata-rata selisih gol, atau rekor lima laga terakhir. Fitur dihitung hanya dari laga sebelum tanggal pertandingan supaya informasi masa depan tidak bocor.
- Pembagian data menurut waktu: latih pada laga lama, uji pada laga yang lebih baru.
- Baseline kelas mayoritas selalu dilaporkan di samping akurasi model.

## Menjalankan

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
jupyter notebook Modeling.ipynb
```
