# Laporan Proyek Machine Learning - Rina Rismawati
## Predictive Analytics - Prediksi Kualitas Anggur Merah

---

# Domain Proyek

## Latar Belakang
Industri anggur adalah sektor ekonomi yang signifikan secara global, dengan persaingan yang ketat dan konsumen yang semakin sadar akan kualitas produk. Kualitas anggur merah dipengaruhi oleh berbagai faktor fisikokimia yang kompleks, mulai dari tingkat keasaman hingga kandungan alkohol. Menentukan kualitas anggur secara manual oleh pakar (sommelier) bisa menjadi proses yang subjektif, memakan waktu, dan mahal. Oleh karena itu, terdapat kebutuhan untuk mengembangkan metode yang lebih objektif dan efisien untuk memprediksi kualitas anggur.

### Mengapa dan Bagaimana Masalah ini Harus Diselesaikan
Mampu memprediksi kualitas anggur secara akurat berdasarkan karakteristik fisikokimianya akan memberikan beberapa manfaat:
* **Efisiensi Produksi:** Produsen anggur dapat mengidentifikasi faktor-faktor kunci yang memengaruhi kualitas, memungkinkan mereka untuk menyesuaikan proses produksi dan meningkatkan hasil panen anggur berkualitas tinggi.
* **Kontrol Kualitas:** Membantu dalam proses kontrol kualitas, memungkinkan identifikasi dini batch anggur yang mungkin memiliki kualitas di bawah standar sebelum dipasarkan.
* **Pengurangan Biaya:** Mengurangi ketergantungan pada penilaian manual yang mahal, sehingga menghemat biaya operasional.
* **Keunggulan Kompetitif:** Produsen dapat lebih konsisten menghasilkan anggur berkualitas, yang pada akhirnya dapat meningkatkan reputasi dan daya saing di pasar.

Pendekatan machine learning sangat cocok untuk masalah ini karena kualitas anggur adalah fungsi dari banyak variabel input yang saling berinteraksi, dan hubungan ini mungkin tidak linier atau mudah dijelaskan dengan aturan sederhana. Model machine learning dapat belajar pola-pola kompleks dari data fisikokimia untuk memprediksi kualitas anggur secara objektif.

### Hasil Riset Terkait atau Referensi
Dataset yang digunakan dalam proyek ini umumnya berasal dari penelitian atau publikasi yang menganalisis korelasi antara properti fisikokimia dan kualitas anggur yang dinilai oleh pakar. Referensi utama untuk dataset semacam ini sering ditemukan di repositori data machine learning seperti [UCI Machine Learning Repository - Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality).

---

# Business Understanding

## Problem Statements (Pernyataan Masalah)
1.  Bagaimana karakteristik fisikokimia (seperti `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`) memengaruhi kualitas anggur merah?
2.  Dapatkah kita membangun sebuah model machine learning yang dapat memprediksi kualitas anggur merah berdasarkan properti fisikokimia yang diukur?

## Goals (Tujuan)
1.  Mengidentifikasi fitur-fitur fisikokimia kunci yang memiliki korelasi signifikan dengan kualitas anggur merah.
2.  Mengembangkan dan mengevaluasi model regresi machine learning yang mampu memprediksi skor kualitas anggur merah secara akurat berdasarkan fitur-fitur fisikokimia.

## Solution Statement
Untuk mencapai tujuan ini, kami akan mengimplementasikan dua pendekatan machine learning:
1.  **Pendekatan Regresi (Baseline Model):** Menggunakan algoritma regresi dasar seperti **Linear Regression** atau **Decision Tree Regressor** untuk memprediksi nilai kualitas anggur secara kontinyu. Model ini akan berfungsi sebagai dasar perbandingan kinerja.
2.  **Pendekatan Regresi Lanjutan (Improvement Model):** Menggunakan algoritma regresi yang lebih canggih seperti **Support Vector Regressor (SVR)** atau **Gradient Boosting Regressor** untuk memprediksi nilai kualitas anggur secara kontinyu. Selain itu, *hyperparameter tuning* akan dilakukan pada model regresi ini untuk mengoptimalkan kinerjanya. Pendekatan regresi ini diharapkan dapat memberikan prediksi yang lebih granular, memungkinkan pengukuran kinerja model dengan metrik evaluasi seperti Mean Absolute Error (MAE) atau Root Mean Squared Error (RMSE).

---

# Data Understanding

Dataset yang digunakan adalah `winequality-red.csv`. Ini adalah dataset kuantitatif yang mengandung berbagai properti fisikokimia dari sampel anggur merah dan skor kualitas yang dinilai oleh pakar. Dataset ini memenuhi kriteria minimum 500 sampel data.

## Tautan Sumber Data
Dataset ini diambil dari:
[UCI Machine Learning Repository - Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)

## Uraian Seluruh Variabel atau Fitur pada Data
Dataset ini memiliki 11 fitur input yang merupakan properti fisikokimia anggur merah, dan 1 fitur output (target) yang mewakili kualitas anggur:

**Fitur Input (Numerik):**
* `fixed acidity`: (asam tetap) Sebagian besar asam yang terlibat dengan anggur, atau asam non-volatil.
* `volatile acidity`: (keasaman volatil) Jumlah asam asetat dalam anggur, yang pada tingkat tinggi dapat menyebabkan rasa cuka yang tidak menyenangkan.
* `citric acid`: (asam sitrat) Asam yang ditemukan dalam jumlah kecil, dapat menambah 'kesegaran' dan rasa pada anggur.
* `residual sugar`: (gula sisa) Jumlah gula yang tersisa setelah fermentasi berhenti. Jarang ditemukan anggur di bawah 1 gram/liter dan anggur di atas 45 gram/liter.
* `chlorides`: (klorida) Jumlah garam dalam anggur.
* `free sulfur dioxide`: (sulfur dioksida bebas) Bagian SO2 yang tidak terikat, yang mencegah pertumbuhan mikroba dan oksidasi anggur.
* `total sulfur dioxide`: (total sulfur dioksida) Jumlah bentuk bebas dan terikat dari SO2.
* `density`: (densitas) Kepadatan anggur, berhubungan dengan kandungan alkohol dan gula.
* `pH`: Menggambarkan seberapa asam atau basa anggur pada skala 0 (sangat asam) hingga 14 (sangat basa). Kebanyakan anggur berada di antara 3-4 pada skala pH.
* `sulphates`: (sulfat) Sebuah aditif anggur yang dapat berkontribusi pada tingkat gas sulfur dioksida, yang bertindak sebagai agen antimikroba dan antioksidan.
* `alcohol`: (alkohol) Kandungan alkohol dalam anggur.

**Fitur Output (Target - Numerik, diskrit):**
* `quality`: (kualitas) Skor kualitas anggur (berdasarkan data, ini adalah nilai diskrit, biasanya dari 3 hingga 8).

### Tahapan untuk Memahami Data (Exploratory Data Analysis / EDA)
Untuk memahami data lebih dalam, beberapa tahapan *exploratory data analysis* (EDA) telah dilakukan:
* **Inspeksi Awal:** Melihat informasi dasar seperti jumlah baris, kolom, tipe data, dan statistik deskriptif.
* **Pengecekan Kualitas Data:** Mengidentifikasi *missing values* dan duplikasi data.
* **Analisis Distribusi Variabel Target:** Memahami sebaran skor kualitas anggur.
* **Analisis Korelasi:** Memvisualisasikan hubungan antar fitur menggunakan heatmap.

Berikut adalah visualisasi distribusi skor kualitas anggur dalam dataset:
![Distribusi Variabel Target](https://raw.githubusercontent.com/Rinarsm/dicoding_predictive_analytics/main/Distribusi%20Variabel%20Target%20(Quality).png)

Untuk memahami hubungan antar fitur, berikut adalah heatmap korelasi:
![Heatmap Correlation Matrix](https://raw.githubusercontent.com/Rinarsm/dicoding_predictive_analytics/main/Heatmap%20korelasi.png)

### Ringkasan Data Understanding:
* Dataset memiliki 1599 baris dan 12 kolom.
* Semua kolom memiliki tipe data numerik (float64 atau int64).
* Tidak ada nilai yang hilang (missing values).
* Terdapat 240 baris duplikat yang akan dihapus pada tahap persiapan data.
* Variabel target 'quality' memiliki rentang nilai 3 hingga 8, dengan sebagian besar data berada pada kualitas 5 dan 6.
* Beberapa fitur menunjukkan korelasi yang jelas dengan kualitas, terutama 'alcohol' (positif) dan 'volatile acidity' (negatif).

---

# Data Preparation

## Teknik Data Preparation yang Dilakukan:
1.  **Penghapusan Duplikasi Data:** Menghapus baris-baris yang merupakan duplikat penuh untuk menghindari bias dalam pelatihan model.
    * **Penjelasan Proses:** Baris duplikat diidentifikasi dan dihapus dari dataset.
    * **Alasan:** Duplikasi data dapat menyebabkan model belajar pola yang salah atau terlalu optimis dalam evaluasi, karena data yang sama akan muncul di training dan testing set. Hal ini dapat menimbulkan *overfitting* yang tidak representatif.
2.  **Pemilihan Fitur dan Target:** Memisahkan fitur-fitur independen (X) dari variabel target (y).
    * **Penjelasan Proses:** Kolom 'quality' diisolasi sebagai variabel target (`y`), sementara semua kolom lainnya dianggap sebagai fitur (`X`).
    * **Alasan:** Ini adalah langkah fundamental dalam machine learning untuk mendefinisikan apa yang akan diprediksi (target) dan apa yang akan digunakan untuk membuat prediksi (fitur).
3.  **Pembagian Data:** Membagi dataset menjadi data pelatihan (training set) dan data pengujian (testing set) untuk evaluasi model yang tidak bias.
    * **Penjelasan Proses:** Dataset dibagi menjadi 80% data pelatihan dan 20% data pengujian menggunakan fungsi `train_test_split` dari scikit-learn dengan `random_state` yang tetap untuk reproduksibilitas.
    * **Alasan:** Pembagian data adalah praktik standar untuk mengevaluasi kinerja model pada data yang belum pernah dilihat sebelumnya. Ini membantu mengukur seberapa baik model dapat menggeneralisasi ke data baru.
4.  **Feature Scaling:** Menerapkan standardisasi pada fitur-fitur numerik untuk memastikan bahwa tidak ada fitur yang mendominasi yang lain karena skala nilai yang berbeda. Ini sangat penting untuk algoritma yang sensitif terhadap skala data seperti SVR.
    * **Penjelasan Proses:** Fitur-fitur numerik distandardisasi menggunakan `StandardScaler`. Proses `fit_transform` diterapkan pada data pelatihan, dan `transform` diterapkan pada data pengujian.
    * **Alasan:** Fitur-fitur dalam dataset ini memiliki rentang nilai yang bervariasi (misalnya, pH dari 2-4, total sulfur dioxide hingga 300). Standardisasi (mengubah data menjadi rata-rata 0 dan standar deviasi 1) membantu algoritma machine learning (terutama yang berbasis jarak seperti SVR) untuk belajar lebih efektif dan mencegah fitur dengan skala besar mendominasi perhitungan.

### Ringkasan Data Preparation:
* Dataset dibersihkan dari 240 baris duplikat, menghasilkan 1359 sampel unik.
* Data dibagi menjadi fitur (`X`) dan target (`y`).
* Data dibagi menjadi set pelatihan (80%, 1087 sampel) dan set pengujian (20%, 272 sampel).
* Fitur-fitur numerik distandardisasi menggunakan `StandardScaler` untuk memastikan skala yang seragam, yang penting untuk kinerja model regresi.

---

# Modeling

## Tahapan dan Parameter yang Digunakan pada Proses Pemodelan:

### Model 1: Linear Regression (Baseline Model)
* **Algoritma:** Linear Regression.
* **Cara Kerja:** Linear Regression adalah algoritma regresi dasar yang memodelkan hubungan antara variabel dependen (target) dan satu atau lebih variabel independen (fitur) sebagai garis lurus. Tujuannya adalah menemukan koefisien (bobot) untuk setiap fitur yang meminimalkan jumlah kuadrat *error* (selisih antara nilai prediksi dan nilai aktual).
* **Parameter:** Tidak ada parameter spesifik yang di-tuning secara ekstensif pada model dasar ini.
* **Kelebihan:** Sederhana, cepat dilatih, mudah diinterpretasi.
* **Kekurangan:** Asumsi hubungan linier, sensitif terhadap *outliers*, tidak menangkap kompleksitas hubungan non-linier.

### Model 2: Support Vector Regressor (SVR - Improvement Model)
* **Algoritma:** Support Vector Regressor.
* **Cara Kerja:** SVR adalah perpanjangan dari Support Vector Machine (SVM) untuk masalah regresi. Alih-alih mencoba memfit garis ke data dengan meminimalkan *error* kuadrat, SVR mencoba menemukan *hyperplane* yang paling cocok dengan margin *error* yang dapat diterima (epsilon-insensitive tube). Ini berarti SVR mengabaikan *error* yang lebih kecil dari epsilon dan fokus pada *error* di luar batas tersebut, sehingga lebih *robust* terhadap *outliers*. Dengan penggunaan *kernel trick*, SVR juga dapat menangani hubungan non-linier dalam data.
* **Parameter:** Parameter `C`, `gamma`, dan `kernel` akan di-tuning menggunakan `GridSearchCV`.
    * `C`: Regularization parameter. Mengontrol *trade-off* antara kelancaran garis regresi dan toleransi terhadap *error*. Semakin kecil nilai C, semakin besar regularisasi (toleransi *error* lebih tinggi).
    * `gamma`: Kernel coefficient untuk 'rbf', 'poly', dan 'sigmoid'. Mendefinisikan seberapa jauh pengaruh satu sampel pelatihan tunggal. Nilai gamma yang besar berarti pengaruh terbatas pada sampel yang dekat.
    * `kernel`: Menentukan fungsi kernel yang akan digunakan dalam algoritma. Pilihan umum termasuk 'linear', 'poly', 'rbf', dan 'sigmoid'.
* **Kelebihan:** Efektif dalam ruang berdimensi tinggi, bekerja baik untuk data yang tidak linier dengan penggunaan kernel.
* **Kekurangan:** Waktu pelatihan yang lambat pada dataset besar, sensitif terhadap *scaling*, dan sulit diinterpretasi.

### Model 3: Gradient Boosting Regressor (Improvement Model)
* **Algoritma:** Gradient Boosting Regressor.
* **Cara Kerja:** Gradient Boosting adalah teknik *ensemble* yang membangun model secara sekuensial. Ini adalah metode yang kuat yang menggabungkan banyak "pembelajar lemah" (biasanya pohon keputusan dangkal) untuk membentuk satu "pembelajar kuat". Pada setiap iterasi, model baru ditambahkan untuk memperbaiki *error* (residual) dari model sebelumnya. Proses ini secara bertahap mengurangi *error* prediksi, bergerak ke arah gradien fungsi *loss*.
* **Parameter:** Parameter `n_estimators`, `learning_rate`, dan `max_depth` akan di-tuning menggunakan `GridSearchCV`.
    * `n_estimators`: Jumlah tahap *boosting* yang akan dilakukan (yaitu, jumlah pohon keputusan yang akan dibangun). Jumlah pohon yang lebih tinggi cenderung meningkatkan akurasi tetapi juga meningkatkan risiko *overfitting*.
    * `learning_rate`: Mengontrol seberapa besar kontribusi setiap pohon baru dalam mengurangi *error*. Nilai yang lebih kecil membutuhkan `n_estimators` yang lebih tinggi tetapi cenderung menghasilkan model yang lebih *robust* dan akurat.
    * `max_depth`: Kedalaman maksimum setiap pohon keputusan individu dalam *ensemble*. Mengontrol kompleksitas setiap pembelajar lemah.
* **Kelebihan:** Seringkali memberikan akurasi tinggi, dapat menangani berbagai jenis data, dan cukup *robust* terhadap *outliers*.
* **Kekurangan:** Cenderung lebih kompleks dan lambat daripada Random Forest, dapat *overfit* jika `n_estimators` terlalu besar atau `learning_rate` terlalu kecil.

### Observasi Awal Model:
Dari evaluasi awal, kita bisa melihat bahwa model Linear Regression memiliki kinerja yang paling rendah (R2 Score terendah, MAE/RMSE tertinggi). SVR dan Gradient Boosting Regressor menunjukkan kinerja yang lebih baik. Selanjutnya, kita akan fokus pada peningkatan kinerja SVR dan Gradient Boosting Regresi melalui *hyperparameter tuning*.

### Proses Improvement yang Dilakukan:
* Untuk SVR, saya melakukan pencarian grid (GridSearchCV) pada parameter `C`, `gamma`, dan `kernel` untuk menemukan kombinasi yang memberikan Mean Absolute Error (MAE) terendah.
* Untuk Gradient Boosting Regressor, saya melakukan pencarian grid pada parameter `n_estimators`, `learning_rate`, dan `max_depth` untuk mengoptimalkan kinerja.
* Tujuan dari *hyperparameter tuning* ini adalah untuk meningkatkan kemampuan generalisasi model dan mengurangi *error* prediksi pada data yang belum terlihat.

---

# Evaluation

## Metrik Evaluasi yang Digunakan dan Penjelasan:

### 1. Mean Absolute Error (MAE)
* **Formula:** $ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $
    * $y_i$: nilai kualitas aktual
    * $\hat{y}_i$: nilai kualitas prediksi
    * $n$: jumlah sampel
* **Penjelasan:** MAE mengukur rata-rata magnitudo *error* dalam satu set prediksi, tanpa mempertimbangkan arahnya. Ini adalah rata-rata absolut dari perbedaan antara prediksi dan observasi aktual. MAE lebih *robust* terhadap *outliers* dibandingkan MSE.

### 2. Root Mean Squared Error (RMSE)
* **Formula:** $ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $
* **Penjelasan:** RMSE adalah akar kuadrat dari rata-rata kuadrat *error*. Metrik ini memberikan bobot lebih pada *error* yang lebih besar karena adanya pengkuadratan. RMSE berada dalam unit yang sama dengan variabel target, sehingga mudah diinterpretasi.

### 3. R-squared (R2 Score)
* **Formula:** $ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} $
    * $\bar{y}$: rata-rata nilai kualitas aktual
* **Penjelasan:** R2 Score (koefisien determinasi) mengukur proporsi varians dalam variabel dependen yang dapat diprediksi dari variabel independen. Nilai R2 berkisar antara 0 hingga 1 (atau bisa negatif jika model sangat buruk). Nilai 1 berarti model menjelaskan semua varians, sedangkan 0 berarti model tidak menjelaskan varians sama sekali. Metrik ini menunjukkan seberapa baik model menyesuaikan dengan data.

## Hasil Proyek Berdasarkan Metrik Evaluasi:
Kami mengevaluasi model terbaik (setelah *hyperparameter tuning*) menggunakan metrik-metrik di atas pada data pengujian.

### Kesimpulan Evaluasi:
Berdasarkan metrik evaluasi pada data pengujian, model **Gradient Boosting Regressor** menunjukkan kinerja terbaik dengan R2 Score **0.4560**, MAE **0.4806**, dan RMSE **0.6208**.

* **R2 Score 0.4560**: Menunjukkan bahwa model mampu menjelaskan sekitar 45.60% dari varians dalam kualitas anggur. Semakin dekat ke 1, semakin baik model menjelaskan variabilitas data.
* **MAE 0.4806**: Ini berarti, rata-rata, prediksi kualitas anggur oleh model menyimpang sekitar 0.4806 poin dari nilai kualitas aktual.
* **RMSE 0.6208**: Memberikan gambaran seberapa besar *error* prediksi, dengan penekanan pada *error* yang lebih besar.

Model ini layak dipilih karena mengungguli model baseline (Linear Regression) secara signifikan dalam hal R2 Score yang lebih tinggi, serta memiliki nilai MAE dan RMSE yang lebih rendah, menunjukkan kemampuan prediksi yang lebih baik dan *error* yang lebih kecil. Performa yang diperoleh setelah *hyperparameter tuning* juga menunjukkan bahwa Gradient Boosting Regressor mampu menangkap pola yang lebih kompleks dalam data untuk menghasilkan prediksi kualitas anggur yang lebih akurat.

# Visualisasi Hasil Prediksi

Untuk mendapatkan pemahaman visual tentang kinerja model, kita dapat memplot nilai aktual versus nilai prediksi.

![Plotting Actual vs Predicted Values for the Best Model](https://raw.githubusercontent.com/Rinarsm/dicoding_predictive_analytics/main/Plotting%20Actual%20vs%20Predicted%20Values%20for%20the%20Best%20Model.png)

![Residual Plot - Histogram](https://raw.githubusercontent.com/Rinarsm/dicoding_predictive_analytics/main/Residual%20Plot.png)

![Residuals vs Predicted Values - Scatter Plot](https://raw.githubusercontent.com/Rinarsm/dicoding_predictive_analytics/main/Residuals%20vs%20Predicted%20Values.png)