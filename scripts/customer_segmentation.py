import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


def main():
    df = pd.read_excel('../data/Online Retail.xlsx')
    print("Data loaded successfully. First 5 rows:")
    print(df.head())


    df = df.dropna(subset=['CustomerID'])
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C')]
    print("Data cleaned. Shape after cleaning:", df.shape)


    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    customer_df = df.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',  # уникальные транзакции
        'Quantity': 'sum',  # количество товаров
        'TotalPrice': 'sum'  # суммарная выручка
    }).rename(columns={'InvoiceNo': 'Frequency', 'Quantity': 'TotalQuantity'})

    print("Feature engineering complete. Customer summary:")
    print(customer_df.head())


    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    customer_df['Cluster'] = kmeans.fit_predict(customer_df)
    print("Clustering completed. Cluster counts:")
    print(customer_df['Cluster'].value_counts())


    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=customer_df, x='Frequency', y='TotalPrice', hue='Cluster', palette='viridis')
    plt.title('Customer Segmentation: Frequency vs TotalPrice')
    plt.xlabel('Frequency (Number of Invoices)')
    plt.ylabel('Total Price')

    plt.savefig('../plots/cluster_scatter_script.png')
    plt.show()

    top_products_list = []

    for cluster in sorted(customer_df['Cluster'].unique()):
        cluster_customer_ids = customer_df[customer_df['Cluster'] == cluster].index
        cluster_df = df[df['CustomerID'].isin(cluster_customer_ids)]
        top_products = cluster_df['Description'].value_counts().head(5).reset_index()
        top_products.columns = ['Product', 'Count']
        top_products['Cluster'] = cluster
        top_products_list.append(top_products)

    top_products_df = pd.concat(top_products_list, ignore_index=True)
    top_products_df.to_csv('../plots/top_products_script.csv', index=False)

    print("Data on the top 5 products for each cluster is saved in '../plots/top_products_script.csv'")


if __name__ == '__main__':
    main()
