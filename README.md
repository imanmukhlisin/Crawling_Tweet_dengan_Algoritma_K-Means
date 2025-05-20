# Clustering Hasil Crawling Tweet

Materi ini bertujuan untuk melakukan **clustering** pada data hasil crawling tweet menggunakan algoritma **K-Means**. Data tweet yang telah diolah akan dikelompokkan ke dalam beberapa cluster berdasarkan kemiripan teksnya.

## Fitur
- **Preprocessing Teks**: Membersihkan teks dari simbol, link, dan stopword bahasa Indonesia.
- **Vektorisasi Teks**: Menggunakan **TF-IDF** untuk mengubah teks menjadi representasi numerik.
- **Clustering**: Mengelompokkan tweet ke dalam cluster menggunakan algoritma **K-Means**.
- **Output CSV**: Menyimpan hasil clustering ke dalam file CSV.

## Prasyarat
Pastikan Anda telah menginstal semua dependensi berikut:
1.**Jupyter Notebooak**
2. **Python 3.x**
3. Modul Python:
   - `pandas`
   - `scikit-learn`
   - `Sastrawi`

### Instalasi Modul
Jalankan perintah berikut untuk menginstal modul yang diperlukan:
```bash
pip install pandas scikit-learn Sastrawi

