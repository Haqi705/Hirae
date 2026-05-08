import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# LOAD DATA
df = pd.read_csv('data_praktikum_analisis_data - data_praktikum_analisis_data.csv')

# CEK DATA
print(df.columns)
print(df.head())
print(df.info())

# HAPUS NULL
df = df.dropna()

# UBAH FORMAT TANGGAL
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# HAPUS HARGA NEGATIF
df = df[df['Price_Per_Unit'] > 0]

# VISUALISASI PRODUK UNDERPERFORMER
plt.figure(figsize=(8,6))

plt.scatter(
    df['Price_Per_Unit'],
    df['Quantity']
)

plt.xlabel('Price Per Unit')
plt.ylabel('Quantity')
plt.title('Produk Underperformer')

plt.show()

# RFM ANALYSIS
snapshot_date = df['Order_Date'].max() + dt.timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'Order_Date': lambda x: (snapshot_date - x.max()).days,
    'Order_ID': 'count',
    'Total_Sales': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

print(rfm.head())

# ANALISIS KATEGORI
category_analysis = df.groupby('Product_Category').agg({
    'Total_Sales': 'sum',
    'Ad_Budget': 'sum'
})

category_analysis['Efficiency'] = (
    category_analysis['Total_Sales']
    /
    category_analysis['Ad_Budget']
)

print(category_analysis)

# VISUALISASI KATEGORI
plt.figure(figsize=(10,6))

plt.barh(
    category_analysis.index,
    category_analysis['Efficiency']
)

plt.xlabel('Efficiency')
plt.ylabel('Category')
plt.title('Efisiensi Kategori')

plt.show()

# UJI HIPOTESIS
median_ad = df['Ad_Budget'].median()

high_ads = df[df['Ad_Budget'] > median_ad]

low_ads = df[df['Ad_Budget'] <= median_ad]

print(high_ads['Total_Sales'].mean())
print(low_ads['Total_Sales'].mean())

# REGRESI
X = df[['Ad_Budget']]

y = df['Total_Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

print(model.coef_[0])
print(model.score(X_test, y_test))