# Laporan Proyek Machine Learning - Muh. Arsan Akbar

## Domain Proyek

Pendidikan merupakan fondasi utama dalam membentuk sumber daya manusia yang kompeten dan berdaya saing tinggi, terutama dalam menghadapi tantangan globalisasi dan kemajuan teknologi. Kualitas pendidikan yang baik menjadi kunci dalam mencetak individu yang produktif dan mampu memberikan kontribusi positif terhadap masyarakat. Namun demikian, sistem pendidikan di berbagai negara, termasuk Indonesia, masih menghadapi tantangan besar, khususnya terkait dengan tingginya angka putus sekolah dan rendahnya tingkat retensi siswa (Tjandra et al., 2022). Permasalahan ini tidak hanya menghambat perkembangan sektor pendidikan, tetapi juga berdampak jangka panjang terhadap masa depan generasi muda.

Salah satu penyebab utama dari tingginya angka putus sekolah adalah rendahnya kinerja akademik siswa. Ketika siswa mengalami kesulitan dalam memahami materi pelajaran dan tidak mencapai hasil belajar yang memadai, mereka lebih rentan untuk kehilangan motivasi dan akhirnya memilih keluar dari sistem pendidikan formal (Gusnina et al., 2022). Dalam konteks ini, diperlukan strategi preventif yang tepat untuk mengidentifikasi siswa yang berisiko dan memberikan intervensi secara dini (Ismanto et al., 2022).

Penerapan teknologi seperti machine learning dalam dunia pendidikan memberikan peluang besar untuk mengatasi permasalahan tersebut. Salah satunya adalah dengan membangun sistem prediksi kinerja akademik siswa berdasarkan data historis dan karakteristik individual. Kinerja akademik sendiri merupakan indikator yang mencerminkan sejauh mana siswa berhasil mencapai tujuan pembelajaran dalam bidang tertentu. Dengan melakukan prediksi terhadap performa akademik, pendidik dapat memberikan bimbingan, sumber daya tambahan, serta intervensi yang disesuaikan dengan kebutuhan masing-masing siswa (Masangu et al., 2020).

Lebih jauh lagi, kinerja akademik yang baik tidak hanya menentukan kesuksesan siswa di lingkungan sekolah, tetapi juga menjadi faktor penting dalam kesiapan memasuki dunia kerja. Siswa yang unggul secara akademik umumnya memiliki peluang yang lebih besar untuk memperoleh pekerjaan berkualitas dan memiliki jenjang karier yang lebih baik (Adane et al., 2023). Oleh karena itu, meningkatkan kinerja akademik siswa memiliki dampak luas, baik dari sisi pendidikan maupun kehidupan profesional mereka setelah lulus.

Berdasarkan urgensi tersebut, proyek ini akan mengembangkan sistem prediksi kinerja akademik siswa menggunakan pendekatan machine learning, dengan memanfaatkan fitur-fitur seperti skor matematika, membaca, menulis, serta variabel sosiodemografis seperti gender, etnis, dan latar belakang pendidikan orang tua. Melalui pendekatan ini, diharapkan sekolah dan lembaga pendidikan dapat secara proaktif mendeteksi potensi permasalahan akademik dan menyediakan solusi berbasis data yang lebih efektif.

Referensi:

[Tjandra, E., Kusumawardani, S. S., & Ferdiana, R. (2022). Student performance prediction in higher education: A comprehensive review. AIP Conference Proceedings. https://doi.org/10.1063/5.0080187](https://doi.org/10.1063/5.0080187)

[Gusnina, M., Wiharto, N., & Salamah, U. (2022). Student performance prediction in Sebelas Maret University based on the Random Forest algorithm. Ingénierie Des Systèmes D Information, 27(3), 495–501. https://doi.org/10.18280/isi.270317]( 
https://doi.org/10.18280/isi.270317)

[Ismanto, E., Ghani, H. A., Saleh, N. I. M., Amien, J. A., & Gunawan, R. (2022). Recent systematic review on student performance prediction using backpropagation algorithms. TELKOMNIKA (Telecommunication Computing Electronics and Control), 20(3), 597. https://doi.org/10.12928/telkomnika.v20i3.21963]( 
http://doi.org/10.12928/telkomnika.v20i3.21963)

[Adane, M. D., Deku, J. K., & Asare, E. K. (2023). Performance analysis of machine learning algorithms in prediction of student academic performance. Journal of Advances in Mathematics and Computer Science, 38(5), 74–86. https://doi.org/10.9734/jamcs/2023/v38i51762]( 
https://doi.org/10.9734/jamcs/2023/v38i51762)

## Business Understanding

Pendidikan adalah pilar utama pembangunan sumber daya manusia. Namun, tantangan serius masih dihadapi, terutama terkait dengan rendahnya kinerja akademik siswa yang dapat memicu putus sekolah. Oleh karena itu, pemanfaatan pendekatan prediktif berbasis machine learning dalam dunia pendidikan menjadi langkah yang strategis untuk mendeteksi potensi permasalahan akademik sejak dini.

### Problem Statements
- Bagaimana memprediksi kinerja akademik siswa secara akurat berdasarkan data karakteristik tertentu?
- Fitur apa saja yang paling signifikan dan berpengaruh dalam menentukan performa akademik siswa?

### Goals
- Membangun model prediksi machine learning yang mampu memperkirakan kinerja akademik siswa secara akurat berdasarkan fitur-fitur yang tersedia.
- Mengidentifikasi fitur-fitur yang paling berkorelasi dan berkontribusi signifikan terhadap keberhasilan akademik siswa.

### Solution statements
- Menerapkan algoritma KNeighborsRegressor, RandomForestRegressor dan AdaBoostRegressor untuk membuat model prediksi performa siswa
- Membuat fitur baru dari fitur yang ada (feature engineering), yaitu menggabungkan skor rata-rata dari tiga indikator penilaian utama untuk membentuk sebuah label yang baru. Fitur tersebut nantinya akan digunakan sebagai acuan untuk menilai fitur yang paling signifikan terhadap keberhasilan siswa.
- Menghitung Mean Squared Error masing-masing algoritma pada data train dan test untuk mencari model yang terbaik
    
## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Student Performance Prediction Dataset yang bersumber dari platform [kaggle](https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics/data). Dataset ini berisi informasi mengenai performa siswa sekolah menengah atas di Amerika Serikat dalam tiga mata pelajaran utama: matematika, membaca, dan menulis. Selain skor akademik, dataset ini juga mencakup berbagai fitur demografis seperti jenis kelamin, latar belakang etnis, tingkat pendidikan orang tua, status subsidi makan siang, dan apakah siswa telah mengikuti kursus persiapan ujian atau tidak.

### Variabel-variabel pada dataset adalah sebagai berikut:
- gender: Merupakan jenis kelamin siswa, dengan nilai male (laki-laki) atau female (perempuan).

- race/ethnicity: Menunjukkan latar belakang ras atau etnis siswa. Kategori ini dibagi menjadi lima kelompok yang mengacu pada klasifikasi di AS (Group A sampai Group E).

- parental level of education: Menyatakan tingkat pendidikan tertinggi yang dicapai oleh orang tua atau wali siswa. Nilainya meliputi: some high school, high school, some college, associate’s degree, bachelor’s degree, dan master’s degree.

- lunch: Menunjukkan apakah siswa menerima makan siang secara gratis atau dengan harga subsidi (free/reduced) atau membayar penuh (standard).

- test preparation course: Menunjukkan apakah siswa telah menyelesaikan kursus persiapan ujian sebelum mengikuti tes standar. Nilainya adalah completed atau none.

- math score: Skor siswa dalam ujian standar mata pelajaran matematika. Nilainya berupa angka dari 0 hingga 100.

- reading score: Skor siswa dalam ujian standar mata pelajaran membaca. Nilainya berupa angka dari 0 hingga 100.

- writing score: Skor siswa dalam ujian standar mata pelajaran menulis. Nilainya berupa angka dari 0 hingga 100.

### Exploratory data analysis - Univariate Analysis
### Fitur-fitur Kategori
### Distribusi Gender
| Gender | Count | Percent |
|--------|-------|---------|
| Male   | 508   | 50.8%   |
| Female | 492   | 49.2%   |

Jumlah siswa laki-laki (50.8%) dan perempuan (49.2%) hampir seimbang. Hal ini menunjukkan tidak adanya bias signifikan dalam representasi gender

### Distribusi Race/Ethnicity
| Race/Ethnicity | Count | Percent |
|----------------|-------|---------|
| Group C        | 323   | 32.3%   |
| Group D        | 257   | 25.7%   |
| Group B        | 198   | 19.8%   |
| Group E        | 143   | 14.3%   |
| Group A        | 79    | 7.9%    |

Sebagian besar siswa berasal dari Group C (32.3%), diikuti oleh Group D (25.7%) dan Group B (19.8%). Group A dan E relatif lebih sedikit.

### Distribusi Parental Level of Education
| Parental Level of Education | Count | Percent |
|-----------------------------|-------|---------|
| Some college                | 224   | 22.4%   |
| High school                 | 215   | 21.5%   |
| Associate's degree          | 204   | 20.4%   |
| Some high school            | 177   | 17.7%   |
| Bachelor's degree           | 105   | 10.5%   |
| Master's degree             | 75    | 7.5%    |

Mayoritas orang tua siswa memiliki tingkat pendidikan “some college” (22.4%), diikuti oleh “high school” (21.5%) dan “associate's degree” (20.4%). Sementara itu, hanya sebagian kecil orang tua yang memiliki gelar “master’s degree” (7.5%). 

### Distribusi Lunch
| Lunch Type     | Count | Percent |
|----------------|-------|---------|
| Standard       | 660   | 66.0%   |
| Free/Reduced   | 340   | 34.0%   |

Sebanyak 66.0% siswa mendapatkan makan siang standar, sedangkan 34.0% menerima makan siang gratis atau diskon.

### Distribusi Test Preparation Course
| Test Preparation Course | Count | Percent |
|-------------------------|-------|---------|
| None                    | 656   | 65.6%   |
| Completed               | 344   | 34.4%   |

Sebanyak 65.6% siswa tidak mengikuti kursus persiapan ujian, sedangkan 34.4% mengikuti.

### Fitur-fitur Numerikal
![EDA Unvariate](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Unvariate%20Numerical.png)

Berdasarkan hasil eksplorasi data univariat terhadap fitur numerik yaitu math score, reading score, dan writing score, dapat disimpulkan bahwa distribusi ketiga skor tersebut cenderung mengikuti pola distribusi normal, meskipun terlihat sedikit condong ke kiri (left-skewed), khususnya pada skor matematika dan menulis. Sebagian besar nilai berada dalam kisaran 60 hingga 80, yang menunjukkan bahwa mayoritas siswa memiliki performa akademik yang cukup baik. Skor membaca menunjukkan distribusi yang paling simetris, serta memiliki konsentrasi nilai tinggi lebih banyak dibanding dua skor lainnya, mengindikasikan bahwa kemampuan membaca siswa secara umum lebih unggul. Sementara itu, skor matematika memiliki beberapa nilai rendah yang secara realita tidak dapat dianggap sebagai outlier, namun tidak terlalu signifikan. Secara keseluruhan, ketiga skor ini menunjukkan distribusi yang baik dan stabil.

### Exploratory data analysis - Multivariate Analysis
### Fitur-fitur Kategorical Terhadap Target
![EDA Multivariate Kategorical](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Multivariate%20Kategorical%20Gender.png)
Siswa perempuan memiliki skor rata-rata sedikit lebih tinggi (±70) dibandingkan siswa laki-laki (±68). Karena selisihnya kecil, maka fitur ini memiliki pengaruh yang rendah terhadap rata-rata skor.

![EDA Multivariate Kategorical](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Multivariate%20Kategorical%20Lunch.png)
Siswa yang mendapatkan makan siang standar memiliki skor rata-rata lebih tinggi (±72) dibandingkan dengan siswa yang mendapatkan makan siang gratis (±64). Hal ini menunjukkan bahwa status makan siang memiliki pengaruh yang cukup kuat terhadap rata-rata skor.

![EDA Multivariate Kategorical](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Multivariate%20Kategorical%20Pendidikan%20Orangtua.png)
Siswa dengan orang tua berpendidikan tinggi seperti bachelor’s degree dan master’s degree cenderung memiliki rata-rata skor lebih tinggi (±71), sedangkan yang berasal dari orang tua dengan pendidikan some high school memiliki rata-rata skor lebih rendah (±65). Meskipun terlihat ada tren, perbedaan antar kelompok tidak terlalu tajam, sehingga fitur ini tidak terlalu berpengaruh terhadap skor.

![EDA Multivariate Kategorical](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Multivariate%20Kategorical%20Persiapan%20Tes.png)
Siswa yang menyelesaikan kursus persiapan tes memiliki skor rata-rata lebih tinggi (±74) dibandingkan yang tidak mengikuti kursus (±67). Hal ini menunjukkan bahwa kursus persiapan tes memiliki pengaruh yang cukup kuat terhadap peningkatan skor rata-rata.

![EDA Multivariate Kategorical](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Multivariate%20Kategorical%20Ras.png)
Kelompok E memiliki skor rata-rata tertinggi (±76), sementara kelompok lain berkisar antara 67 hingga 71. Perbedaan ini menunjukkan adanya variasi, namun tidak konsisten meningkat atau menurun antar kelompok, sehingga fitur ini memiliki pengaruh yang rendah terhadap skor.

### Fitur-fitur Numerical Terhadap Target
![EDA Multivariate Numerical](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Multivariate%20Numerical.png)
Fitur numerik menunjukkan hubungan linear yang kuat satu sama lain, dengan korelasi tertinggi antara reading score dan writing score. Average score sangat bergantung secara proporsional pada ketiga skor asli, dan hubungan linear ini memvalidasi penggunaannya sebagai metrik gabungan. 

## Data Preparation
- Feature Engineering: Membuat fitur baru berupa rata rata score yang didapatkan dari hasil math score + reading score + writing score dibagi 3, hal tersebut dilakukan mengingat belum adanya target pada dataset, sehingga perlu dilakukan feature engineering untuk menghasilkan fitur baru yang relevan.

- Encoding fitur kategori: Feature encoding kategori seperti OneHotEncoder penting dilakukan karena sebagian besar algoritma machine learning tidak dapat menangani data kategorikal secara langsung. Mereka memerlukan input berupa angka, sedangkan beberapa fitur pada dataset tersebut berbentuk kategori, fitur fitur tersebut adalah gender, ras, level pendidikan orangtua, tipe makan siang, dan tes persiapan.

- Reduksi dimensi dengan PCA: Reduksi dimensi dengan PCA (Principal Component Analysis) diperlukan karena fitur math score, reading score, dan writing score menunjukkan korelasi tinggi satu sama lain, yang berarti terdapat redundansi informasi. PCA membantu menyederhanakan fitur-fitur tersebut menjadi beberapa komponen utama yang tetap mempertahankan sebagian besar informasi, sehingga dapat meningkatkan efisiensi model, mengurangi risiko overfitting, dan mempermudah visualisasi data. Selain itu, PCA juga membantu menghilangkan noise dan menjaga struktur data dalam dimensi yang lebih rendah. Oleh karena itu fitur math score, reading score, dan writing score dimasukkan kedalam proses PCA menjadi sebuah fitur yang bernama student performance.

- Train dan test split: Train-test split perlu dilakukan untuk mengevaluasi kinerja model secara objektif. Dengan membagi data menjadi data latih (train) dan data uji (test), kita dapat melatih model pada satu bagian data dan mengujinya pada data yang belum pernah dilihat sebelumnya. Hal ini penting untuk menilai kemampuan generalisasi model terhadap data baru dan mencegah overfitting, yaitu kondisi di mana model terlalu baik dalam menghafal data latih namun buruk dalam memprediksi data baru. Dalam kasus ini, data dibagi 90% untuk pelatihan dan 10% untuk pengujian, memberikan cukup data untuk pembelajaran sambil tetap menyisakan data yang representatif untuk evaluasi.

- Standarisasi: hal tersebut perlu dilakukan untuk menyamakan skala fitur numerik agar model machine learning dapat bekerja secara optimal. Fitur seperti student performance mungkin memiliki rentang nilai yang berbeda dibanding fitur lain, dan ini bisa menyebabkan model lebih condong atau berat sebelah terhadap fitur dengan nilai besar. Dengan standarisasi menggunakan StandardScaler, data diubah agar memiliki rata-rata 0 dan standar deviasi 1, sehingga semua fitur berada dalam skala yang seimbang. Ini sangat penting terutama untuk algoritma yang sensitif terhadap skala data seperti KNN, SVM, dan regresi linier. Hasil standarisasi menunjukkan bahwa data telah terpusat di sekitar nol dengan penyebaran standar yang seragam, memastikan proses pelatihan model menjadi lebih stabil dan akurat.

## Modeling
Pada tahap ini dilakukan pengembangan model machine learning untuk memprediksi skor rata-rata siswa berdasarkan fitur-fitur input yang telah diproses sebelumnya. Tiga algoritma regresi digunakan, yaitu K-Nearest Neighbors (KNN), Random Forest Regressor, dan AdaBoost Regressor.

**K-Nearest Neighbors (KNN)**

Model KNN digunakan dengan parameter n_neighbors=10. Kelebihan KNN adalah sederhana dan tidak membutuhkan proses pelatihan yang kompleks. Namun, KNN sangat sensitif terhadap skala fitur dan kurang efisien pada dataset besar. Model ini menghasilkan MSE (mean squared error) sebesar 0.0137 (train) dan 0.0113 (test).

**Random Forest**

Random Forest digunakan dengan n_estimators=50 dan max_depth=16. Algoritma ini kuat terhadap overfitting dan dapat menangani data nonlinear dengan baik. Hasil evaluasi menunjukkan performa terbaik dibanding model lain, dengan MSE sangat kecil yaitu 0.000009 (train) dan 0.000008 (test). Ini menunjukkan model sangat akurat dalam menangkap pola data.

**AdaBoost Regressor**

AdaBoost digunakan dengan learning_rate=0.05. Algoritma ini bekerja dengan menggabungkan beberapa model lemah (weak learners) secara iteratif untuk meningkatkan performa prediksi. Hasil yang diperoleh cukup baik, dengan MSE 0.0024 (train) dan 0.0029 (test), namun masih kalah dari Random Forest.

Berdasarkan hasil evaluasi MSE pada data latih dan uji, Random Forest dipilih sebagai model terbaik karena menghasilkan error paling rendah di antara semua model yang diuji. Selain itu, model ini juga lebih stabil dan mampu menangani kompleksitas data tanpa mengalami overfitting.

## Evaluation
Karena proyek ini merupakan kasus regresi, maka metrik evaluasi yang digunakan adalah Mean Squared Error (MSE). MSE mengukur rata-rata kuadrat selisih antara nilai aktual (y_true) dan nilai prediksi (y_pred). Semakin kecil nilai MSE, semakin akurat model dalam melakukan prediksi. Metrik ini cocok digunakan karena memberikan penalti yang lebih besar pada kesalahan prediksi yang jauh dari nilai sebenarnya. MSE dihitung menggunakan rumus berikut:

![MSE Formula](https://cdn.analyticsvidhya.com/wp-content/uploads/2024/07/image-37.png)


Cara kerja MSE adalah dengan menghitung selisih antara nilai aktual dan prediksi untuk setiap data, lalu mengkuadratkan selisih tersebut agar tidak ada nilai negatif dan memberi penalti lebih besar terhadap kesalahan prediksi yang jauh. Kemudian seluruh kuadrat error dijumlahkan dan dirata-ratakan.

MSE efektif digunakan dalam regresi karena memberikan pemahaman seberapa besar rata-rata kesalahan model dalam satuan kuadrat dari target. Nilai MSE yang lebih rendah menunjukkan model yang lebih baik dalam memprediksi target.

**Hasil Evaluasi Model**

Berdasarkan hasil evaluasi terhadap tiga model, diperoleh hasil sebagai berikut:
| Model        | Train MSE | Test MSE |
|--------------|-----------|----------|
| KNN          | 0.0137    | 0.0113   |
| RandomForest | 0.000009  | 0.000008 |
| Boosting     | 0.0024    | 0.0029   |


Dari tabel di atas, terlihat bahwa Random Forest Regressor memiliki performa terbaik dengan nilai MSE terkecil baik pada data latih maupun data uji. Hal ini menunjukkan bahwa model ini mampu melakukan generalisasi dengan sangat baik, serta minim terhadap overfitting.

**Evaluasi Prediksi**

Untuk melihat kualitas prediksi lebih lanjut, dilakukan perbandingan antara nilai aktual (y_true) dan hasil prediksi dari ketiga model:
| y_true | KNN  | RandomForest | Boosting |
|--------|------|--------------|----------|
| 56.3   | 60.3 | 56.3         | 56.7     |
| 92.0   | 87.2 | 92.0         | 93.2     |
| 72.0   | 73.5 | 72.0         | 70.4     |
| 63.3   | 67.8 | 63.3         | 63.8     |

Dari tabel tersebut, dapat dilihat bahwa hasil prediksi Random Forest paling konsisten mendekati nilai sebenarnya dibanding model lainnya. Hal ini menguatkan alasan pemilihan Random Forest sebagai model akhir.
