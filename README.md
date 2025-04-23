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

Dataset yang digunakan dalam proyek ini adalah Student Performance Prediction Dataset yang bersumber dari platform [Kaggle](https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics/data).

**Jumlah Data:** Dataset ini terdiri dari 1000 baris (sampel siswa) dan 8 kolom (fitur).

**Kondisi Data:** Berdasarkan analisis awal, dataset ini memiliki kondisi sebagai berikut:
- **Missing Value:** Tidak terdapat nilai yang hilang (*missing value*) dalam dataset.
- **Outlier:** Berdasarkan visualisasi distribusi fitur numerik, tidak terdeteksi adanya *outlier* ekstrem yang secara signifikan dapat mengganggu analisis atau pemodelan. Beberapa nilai ekstrem rendah pada skor matematika mungkin ada, namun dianggap sebagai variasi alami dalam performa siswa.

**Tautan Sumber Data:** [https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics/data](https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics/data)

**Uraian Fitur:**
- `gender`: Merupakan jenis kelamin siswa, dengan nilai `male` (laki-laki) atau `female` (perempuan).
- `race/ethnicity`: Menunjukkan latar belakang ras atau etnis siswa. Kategori ini dibagi menjadi lima kelompok: `group A`, `group B`, `group C`, `group D`, dan `group E`.
- `parental level of education`: Menyatakan tingkat pendidikan tertinggi yang dicapai oleh orang tua atau wali siswa. Nilainya meliputi: `some high school`, `high school`, `some college`, `associate's degree`, `bachelor's degree`, dan `master's degree`.
- `lunch`: Menunjukkan status subsidi makan siang siswa. Nilainya adalah `standard` (membayar penuh) atau `free/reduced` (gratis atau subsidi).
- `test preparation course`: Menunjukkan apakah siswa telah menyelesaikan kursus persiapan ujian. Nilainya adalah `completed` atau `none`.
- `math score`: Skor siswa dalam ujian standar mata pelajaran matematika. Nilainya berupa bilangan bulat dari 0 hingga 100.
- `reading score`: Skor siswa dalam ujian standar mata pelajaran membaca. Nilainya berupa bilangan bulat dari 0 hingga 100.
- `writing score`: Skor siswa dalam ujian standar mata pelajaran menulis. Nilainya berupa bilangan bulat dari 0 hingga 100.



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

**Kesimpulan**

Berdasarkan hasil eksplorasi data (EDA) terhadap variabel kategorikal, dapat disimpulkan bahwa distribusi siswa berdasarkan gender cukup seimbang, dengan proporsi laki-laki sebesar 50.8% dan perempuan 49.2%. Dari sisi latar belakang etnis, mayoritas siswa berasal dari group C (32.3%), diikuti oleh group D (25.7%) dan group B (19.8%), sementara group A dan E masing-masing hanya menyumbang 7.9% dan 14.3%. Latar belakang pendidikan orang tua menunjukkan bahwa sebagian besar berasal dari keluarga dengan tingkat pendidikan menengah, seperti “some college” (22.4%) dan “high school” (21.5%). Hanya sedikit orang tua yang memiliki gelar magister (7.5%), yang menunjukkan bahwa sebagian besar siswa mungkin tidak mendapatkan dukungan akademik dari orang tua dengan pendidikan tinggi.

Dari sisi ekonomi, sebanyak 66.0% siswa mendapatkan makan siang standar, sementara 34.0% menerima makan siang gratis atau bersubsidi, yang sering kali menjadi indikator kondisi sosial-ekonomi yang lebih rendah. Selain itu, hanya 34.4% siswa yang telah menyelesaikan kursus persiapan ujian, sedangkan 65.6% lainnya tidak mengikuti kursus tersebut. Hal ini menunjukkan bahwa sebagian besar siswa mungkin menghadapi keterbatasan dalam akses terhadap persiapan akademik tambahan.

Secara keseluruhan, hasil ini memberikan gambaran bahwa faktor sosial-ekonomi, latar belakang pendidikan orang tua, serta akses terhadap fasilitas belajar tambahan dapat menjadi faktor penting yang memengaruhi performa akademik siswa. Analisis lanjutan sangat dianjurkan untuk melihat bagaimana variabel-variabel ini berkorelasi dengan hasil tes akademik seperti nilai matematika, membaca, dan menulis, guna memperoleh pemahaman yang lebih mendalam dan komprehensif.

### Fitur-fitur Numerikal
![EDA Unvariate](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Unvariate%20Numerical.png)

Berdasarkan hasil eksplorasi data univariat terhadap fitur numerik yaitu math score, reading score, dan writing score, dapat disimpulkan bahwa distribusi ketiga skor tersebut cenderung mengikuti pola distribusi normal, meskipun terlihat sedikit condong ke kiri (left-skewed), khususnya pada skor matematika dan menulis. Sebagian besar nilai berada dalam kisaran 60 hingga 80, yang menunjukkan bahwa mayoritas siswa memiliki performa akademik yang cukup baik. Skor membaca menunjukkan distribusi yang paling simetris, serta memiliki konsentrasi nilai tinggi lebih banyak dibanding dua skor lainnya, mengindikasikan bahwa kemampuan membaca siswa secara umum lebih unggul. Sementara itu, skor matematika memiliki beberapa nilai rendah yang secara realita tidak dapat dianggap sebagai outlier, namun tidak terlalu signifikan. Secara keseluruhan, ketiga skor ini menunjukkan distribusi yang baik dan stabil.

**Kesimpulan**

*   Histogram pertama menampilkan distribusi nilai ujian matematika (math score). Terlihat bahwa distribusi nilai cenderung unimodal dan mendekati distribusi normal, meskipun terdapat sedikit skewness ke kiri (ekor distribusi memanjang ke arah nilai yang lebih rendah). Sebagian besar siswa memperoleh nilai antara 60 hingga 80, dengan puncak frekuensi berada di sekitar nilai 65-70. Terdapat beberapa siswa dengan nilai yang sangat rendah (di bawah 40) dan juga beberapa siswa dengan nilai yang sangat tinggi (di atas 90), namun jumlahnya relatif lebih sedikit dibandingkan dengan kelompok nilai tengah.
*   Histogram kedua menyajikan distribusi nilai ujian membaca (reading score). Distribusi nilai membaca tampak lebih mendekati distribusi normal dibandingkan dengan nilai matematika. Puncak frekuensi berada di sekitar nilai 70-80, dan sebagian besar siswa memperoleh nilai antara 60 hingga 90. Sebaran nilai membaca juga terlihat sedikit lebih lebar dibandingkan dengan nilai matematika, mengindikasikan variasi performa membaca antar siswa yang mungkin lebih besar. Jumlah siswa dengan nilai sangat rendah (di bawah 40) dan sangat tinggi (di atas 95) juga relatif sedikit.
*   Histogram ketiga menggambarkan distribusi nilai ujian menulis (writing score). Distribusi nilai menulis juga terlihat unimodal dengan puncak frekuensi di sekitar nilai 65-75. Sebagian besar siswa memperoleh nilai antara 55 hingga 85. Distribusi nilai menulis menunjukkan skewness negatif yang lebih jelas dibandingkan dengan nilai matematika, dengan ekor distribusi yang lebih panjang ke arah nilai yang lebih rendah. Ini mengindikasikan bahwa terdapat lebih banyak siswa yang memperoleh nilai di bawah rata-rata dibandingkan dengan siswa yang memperoleh nilai jauh di atas rata-rata.


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

**Kesimpulan**

Hasil analisis menunjukkan bahwa variabel kategorikal seperti pendidikan orangtua,jenis kelamin, jenis makan siang, tingkat pendidikan orangtua, partisipasi kursus, dan ras sebenarnya tidak memberikan dampak yang signifikan terhadap nilai rata-rata skor siswa. Hal ini ditunjukkan dengan nilai rata rata skor untuk tiap-tiap variabel yang hanya berada di kisaran 60-75.

### Fitur-fitur Numerical Terhadap Target
![EDA Multivariate Numerical](https://raw.githubusercontent.com/minggo-commits/student-performance-prediction/main/EDA%20Multivariate%20Numerical.png)
Fitur numerik menunjukkan hubungan linear yang kuat satu sama lain, dengan korelasi tertinggi antara reading score dan writing score. Average score sangat bergantung secara proporsional pada ketiga skor asli, dan hubungan linear ini memvalidasi penggunaannya sebagai metrik gabungan. 

**Kesimpulan**

Analisis pair plot ini dengan jelas menunjukkan hubungan linear positif yang sangat kuat antara nilai matematika, membaca, dan menulis dengan rata-rata skor siswa. Selain itu, terdapat korelasi yang tinggi di antara ketiga nilai ujian itu sendiri. Temuan ini menggarisbawahi pentingnya ketiga mata pelajaran ini dalam menentukan performa akademik keseluruhan siswa. Model prediksi rata-rata skor kemungkinan akan sangat akurat jika menggunakan ketiga nilai ujian ini sebagai fitur. Tidak terlihat adanya pola non-linear yang signifikan atau outlier ekstrem yang perlu perhatian khusus dari visualisasi ini.

## Data Preparation

- Karena dataset menunjukkan kondisi yang bersih tanpa adanya nilai yang hilang dan tidak terdapat outlier signifikan yang dapat berdampak negatif, maka tidak dilakukan pengurangan data.

- Feature Engineering: Membuat fitur baru berupa rata rata score yang didapatkan dari hasil math score + reading score + writing score dibagi 3, hal tersebut dilakukan mengingat belum adanya target pada dataset, sehingga perlu dilakukan feature engineering untuk menghasilkan fitur baru yang relevan.

- Encoding fitur kategori: Feature encoding kategori seperti OneHotEncoder penting dilakukan karena sebagian besar algoritma machine learning tidak dapat menangani data kategorikal secara langsung. Mereka memerlukan input berupa angka, sedangkan beberapa fitur pada dataset tersebut berbentuk kategori, fitur fitur tersebut adalah gender, ras, level pendidikan orangtua, tipe makan siang, dan tes persiapan.

- Reduksi dimensi dengan PCA: Reduksi dimensi dengan PCA (Principal Component Analysis) diperlukan karena fitur math score, reading score, dan writing score menunjukkan korelasi tinggi satu sama lain, yang berarti terdapat redundansi informasi. PCA membantu menyederhanakan fitur-fitur tersebut menjadi beberapa komponen utama yang tetap mempertahankan sebagian besar informasi, sehingga dapat meningkatkan efisiensi model, mengurangi risiko overfitting, dan mempermudah visualisasi data. Selain itu, PCA juga membantu menghilangkan noise dan menjaga struktur data dalam dimensi yang lebih rendah. Oleh karena itu fitur math score, reading score, dan writing score dimasukkan kedalam proses PCA menjadi sebuah fitur yang bernama student performance.

- Train dan test split: Train-test split perlu dilakukan untuk mengevaluasi kinerja model secara objektif. Dengan membagi data menjadi data latih (train) dan data uji (test), kita dapat melatih model pada satu bagian data dan mengujinya pada data yang belum pernah dilihat sebelumnya. Hal ini penting untuk menilai kemampuan generalisasi model terhadap data baru dan mencegah overfitting, yaitu kondisi di mana model terlalu baik dalam menghafal data latih namun buruk dalam memprediksi data baru. Dalam kasus ini, data dibagi 90% untuk pelatihan dan 10% untuk pengujian, memberikan cukup data untuk pembelajaran sambil tetap menyisakan data yang representatif untuk evaluasi.

- Standarisasi: hal tersebut perlu dilakukan untuk menyamakan skala fitur numerik agar model machine learning dapat bekerja secara optimal. Fitur seperti student performance mungkin memiliki rentang nilai yang berbeda dibanding fitur lain, dan ini bisa menyebabkan model lebih condong atau berat sebelah terhadap fitur dengan nilai besar. Dengan standarisasi menggunakan StandardScaler, data diubah agar memiliki rata-rata 0 dan standar deviasi 1, sehingga semua fitur berada dalam skala yang seimbang. Ini sangat penting terutama untuk algoritma yang sensitif terhadap skala data seperti KNN, SVM, dan regresi linier. Hasil standarisasi menunjukkan bahwa data telah terpusat di sekitar nol dengan penyebaran standar yang seragam, memastikan proses pelatihan model menjadi lebih stabil dan akurat.

## Modeling
Pada tahap ini dilakukan pengembangan model machine learning untuk memprediksi skor rata-rata siswa berdasarkan fitur-fitur input yang telah diproses sebelumnya. Tiga algoritma regresi digunakan, yaitu K-Nearest Neighbors (KNN), Random Forest Regressor, dan AdaBoost Regressor.

- **K-Nearest Neighbors (KNN)**
  
K-Nearest Neighbors (KNN) adalah algoritma non-parametrik yang bekerja dengan cara membandingkan jarak antara data uji dengan seluruh data latih, lalu memilih k tetangga terdekat untuk melakukan prediksi. Nilai prediksi untuk regresi ditentukan dari rata-rata nilai target dari k tetangga terdekat tersebut. Model KNN digunakan dengan parameter n_neighbors=10 dan untuk parameter lainnya bernilai default. Kelebihan KNN adalah sederhana dan tidak membutuhkan proses pelatihan yang kompleks. Namun, KNN sangat sensitif terhadap skala fitur dan kurang efisien pada dataset besar. Model ini menghasilkan MSE (mean squared error) sebesar 0.0137 (train) dan 0.0113 (test).

- **Random Forest**

Random Forest merupakan algoritma ensemble learning yang menggabungkan banyak pohon keputusan (decision trees) untuk meningkatkan akurasi prediksi. Setiap pohon dilatih pada subset data yang dipilih secara acak (bootstrap), dan hasil prediksi akhir diambil rata-rata dari semua pohon. Random Forest digunakan dengan n_estimators=50, max_depth=16, random_state=55, n_jobs=-1 serta parameter lain yang bernilai default. Algoritma ini mampu menangani data dengan fitur non-linear dan tidak sensitif terhadap skala fitur. Namun random forest memerlukan sumber daya komputasi besar dan kurang interpretatif. Hasil evaluasi menunjukkan performa terbaik dibanding model lain, dengan MSE sangat kecil yaitu 0.000009 (train) dan 0.000008 (test). Ini menunjukkan model sangat akurat dalam menangkap pola data.

- **AdaBoost Regressor**

AdaBoost (Adaptive Boosting) bekerja dengan membentuk model ensemble dari sejumlah weak learners, biasanya decision tree berukuran kecil. Setiap model baru dibangun dengan fokus pada data yang salah diklasifikasikan oleh model sebelumnya. Hasil akhir prediksi merupakan kombinasi tertimbang dari seluruh model. AdaBoost digunakan dengan learning_rate=0.05 dan random_state=55 serta parameter lainnya yang bernilai default. Algoritma ini meningkatkan akurasi model dengan menggabungkan banyak prediktor sederhana, akan tetapi rentan terhadap data outlier dan noise. Hasil yang diperoleh cukup baik, dengan MSE 0.0024 (train) dan 0.0029 (test), namun masih kalah dari Random Forest.

**Kesimpulan**

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
