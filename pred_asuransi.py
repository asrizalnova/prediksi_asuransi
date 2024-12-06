import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open('prediksi_asuransi.sav', 'rb'))

# Load dataset untuk visualisasi dan informasi
df = pd.read_csv('insurance.csv')

# Judul aplikasi
st.title('Prediksi Biaya Medis')

# Dropdown Navbar di Sidebar
menu = st.sidebar.selectbox("Pilih Menu", 
                            ["Home", "Dataset", "Visualisasi", "Prediksi", "Algoritma"])

# Home Page - Deskripsi Aplikasi
if menu == "Home":
    st.header('Deskripsi Aplikasi')
    st.write("""
        Aplikasi ini bertujuan untuk memprediksi biaya asuransi kesehatan berdasarkan beberapa faktor 
        seperti umur, jenis kelamin, BMI, jumlah anak, kebiasaan merokok, dan wilayah tempat tinggal.
    """)
    st.header('Nama Pembuat')
    st.write('Nama : Asrizal Nova Akhsanu')
    st.write('NPM : 223307036')

# Dataset Page - Menampilkan Dataset
elif menu == "Dataset":
    st.header('Dataset')
    st.write('Berikut adalah preview dari dataset yang digunakan untuk melatih model:')
    st.dataframe(df.head())

    st.header('Sumber Dataset')
    st.write('Dataset ini diambil dari [Kaggle: Insurance Dataset](https://www.kaggle.com/code/mariapushkareva/medical-insurance-cost-with-linear-regression/input)')

# Visualisasi Page - Menampilkan Visualisasi
elif menu == "Visualisasi":
    st.header('Visualisasi Dataset')

    # Visualisasi 1: Distribusi Charges
    st.subheader('Distribusi Charges')
    plt.figure(figsize=(8, 5))
    sns.histplot(df['charges'], kde=True, color='blue')
    plt.title('Distribusi Charges')
    plt.xlabel('Charges')
    st.pyplot(plt.gcf())

    # Visualisasi 2: Charges vs Region
    st.subheader('Hubungan Charges dengan Region')
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='region', y='charges', palette='Set2')
    plt.title('Charges vs Region')
    plt.xlabel('Region')
    plt.ylabel('Charges')
    st.pyplot(plt.gcf())

    # Visualisasi 3: BMI vs Charges
    st.subheader('Hubungan BMI dengan Charges')
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', palette='cool')
    plt.title('BMI vs Charges')
    plt.xlabel('BMI')
    plt.ylabel('Charges')
    st.pyplot(plt.gcf())

    # Visualisasi 4: Smoker vs Charges
    st.subheader('Hubungan Smoker dengan Charges')
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='smoker', y='charges', palette='Set1')
    plt.title('Smoker vs Charges')
    plt.xlabel('Smoker')
    plt.ylabel('Charges')
    st.pyplot(plt.gcf())

    # Visualisasi 5: Heatmap Korelasi
    st.subheader('Korelasi Antar Fitur')

    # Copy dataset agar tidak mengubah data asli
    df_encoded = df.copy()

    # Encoding kolom non-numerik
    df_encoded['sex'] = df_encoded['sex'].replace({'male': 1, 'female': 0})
    df_encoded['smoker'] = df_encoded['smoker'].replace({'yes': 1, 'no': 0})
    df_encoded['region'] = df_encoded['region'].replace({
        'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3
    })

    # Plot Heatmap Korelasi
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Matriks Korelasi')
    st.pyplot(plt.gcf())

# Prediksi Page - Input Data dan Prediksi
elif menu == "Prediksi":
    st.subheader('Masukkan Data untuk Prediksi Biaya')

    # Input data
    age = st.number_input('Masukkan Umur (tahun)', min_value=0, max_value=120, step=1, value=None)
    bmi = st.number_input('Masukkan BMI (Body Mass Index(Cara hitung BMI : BB(kg) : TB(m2)))', min_value=0.0, step=0.1, format="%.1f", value=None)
    children = st.number_input('Masukkan Banyak Anak', min_value=0, max_value=10, step=1, value=None)

    # Dropdown untuk jenis kelamin
    sex = st.selectbox(
        'Pilih Jenis Kelamin',
        options=['Perempuan', 'Laki-laki']
    )
    sex_value = 0 if sex == 'Perempuan' else 1

    # Dropdown untuk status merokok
    smoker = st.selectbox(
        'Apakah Anda Merokok?',
        options=['Tidak Merokok', 'Merokok']
    )
    smoker_value = 0 if smoker == 'Tidak Merokok' else 1

    # Dropdown untuk wilayah
    region = st.selectbox(
        'Pilih Wilayah',
        options=['Northeast', 'Northwest', 'Southeast', 'Southwest']
    )
    region_value = {'Northeast': 0, 'Northwest': 1, 'Southeast': 2, 'Southwest': 3}[region]

    # Prediksi
    if st.button('Predict'):
        if age is None or bmi is None or children is None:
            st.error("Harap isi semua data sebelum melakukan prediksi!")
        else:
            predict = model.predict(
                [[age, sex_value, bmi, children, smoker_value, region_value]]
            )
            st.success(f'Prediksi Biaya Asuransi: ${predict[0]:,.2f}')

# Algoritma Page - Menjelaskan Algoritma yang Digunakan
elif menu == "Algoritma":
    st.header('Overview Algoritma yang Digunakan')
    
    # Penjelasan dasar tentang algoritma
    st.subheader('Apa itu Regresi Linear?')
    st.write("""
        Regresi linear adalah salah satu algoritma pembelajaran mesin yang paling sederhana 
        dan paling sering digunakan untuk masalah prediksi kuantitatif. Model ini bekerja dengan
        mencoba menemukan hubungan linear antara variabel independen (fitur) dan variabel dependen (target).
    """)
    
    # Formula dasar regresi linear
    st.subheader('Formula Regresi Linear')
    st.latex(r'''
        y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon
    ''')
    st.write("""
        Di mana:
        - **y**: Nilai yang ingin diprediksi (biaya asuransi dalam kasus ini).
        - **x₁, x₂, ..., xₙ**: Variabel independen (umur, BMI, jumlah anak, dsb.).
        - **β₀**: Intersep atau bias.
        - **β₁, β₂, ..., βₙ**: Koefisien regresi (mengukur pengaruh masing-masing fitur terhadap prediksi).
        - **ε**: Galat atau noise.
    """)

    # Langkah-langkah pembuatan model
    st.subheader('Langkah-langkah Membuat Model')
    st.write("""
        1. **Persiapan Data**:
           - Dataset diolah agar tidak ada nilai kosong atau data tidak valid.
           - Fitur seperti jenis kelamin, status merokok, dan wilayah dikodekan menjadi nilai numerik.
        2. **Split Dataset**:
           - Dataset dibagi menjadi data pelatihan (training) dan data pengujian (testing) untuk menghindari overfitting.
        3. **Pelatihan Model**:
           - Model dilatih menggunakan algoritma regresi linear pada data pelatihan.
        4. **Evaluasi Model**:
           - Model diuji dengan data pengujian menggunakan metrik seperti *Mean Squared Error (MSE)* atau *R² Score* untuk mengukur performa prediksi.
    """)

    # Visualisasi hubungan linear sederhana (contoh visual)
    st.subheader('Visualisasi Hubungan Linear (Contoh)')
    st.write("""
        Hubungan antara BMI dan Biaya Asuransi menunjukkan pola linear sederhana, 
        terutama untuk kategori perokok yang cenderung memiliki biaya lebih tinggi.
    """)
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x='bmi', y='charges', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title('Hubungan BMI dan Charges')
    plt.xlabel('BMI')
    plt.ylabel('Charges')
    st.pyplot(plt.gcf())

    # Kelebihan dan Kekurangan Regresi Linear
    st.subheader('Kelebihan dan Kekurangan Regresi Linear')
    st.write("""
        **Kelebihan**:
        - Sederhana dan mudah dipahami.
        - Cepat dalam pelatihan untuk dataset yang kecil.
        - Interpretasi koefisien regresi jelas.

        **Kekurangan**:
        - Sensitif terhadap outlier.
        - Kurang cocok untuk hubungan yang tidak linear.
        - Performanya menurun jika fitur memiliki multikolinearitas.
    """)
