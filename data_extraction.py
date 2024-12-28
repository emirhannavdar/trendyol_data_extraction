import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from openpyxl import load_workbook
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



def temizle(metin, default_value="Yok"):
    if metin is None or not metin.strip():
        return default_value
    metin = metin.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    while "  " in metin:
        metin = metin.replace("  ", " ")
    return metin.strip()

def temizle_fiyat(fiyat):
    if isinstance(fiyat, str):
        fiyat = fiyat.replace(" TL", "").replace(",", ".").strip()
        try:
            return float(fiyat)
        except ValueError:
            return None
    return None



def veri_cek(base_url, max_products_per_url, headers):
    products_list = []
    product_links = set()
    page = 1

    while len(products_list) < max_products_per_url:
        url = f"{base_url}?pi={page}"
        response = requests.get(url, headers=headers)
        
        print(f"URL: {url}, sayfa: {page}, durum kodu: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Sayfa yüklenemedi: {response.status_code}")
            break
    
        soup = BeautifulSoup(response.content, "html.parser")
        products = soup.find_all("div", class_="p-card-chldrn-cntnr card-border")

        for product in products:

            product_name = product.find("span", class_="prdct-desc-cntnr-name").text if product.find("span", class_="prdct-desc-cntnr-name") else None
            

            price_element = product.find("div", class_="prc-box-dscntd") or product.find("span", class_="prdct-price")
            price = price_element.text if price_element else None
            

            brand = product.find("span", class_="prdct-desc-cntnr-ttl").text if product.find("span", class_="prdct-desc-cntnr-ttl") else None
            

            product_category = base_url  


            product_link_tag = product.find("a", href=True)
            if product_link_tag:
                product_link = product_link_tag['href']
                if product_link.startswith("/"):
                    product_link = "https://www.trendyol.com" + product_link

                if product_link not in product_links:
                    product_links.add(product_link)
                    products_list.append([product_name, price, brand, product_category.split("-")[-4]])

            if len(products_list) >= max_products_per_url:
                break
        
        page += 1
        time.sleep(4)
    
    return products_list


if __name__ == "__main__":

    base_urls = [
        "https://www.trendyol.com/kadin-pantolon-x-g1-c70",
        "https://www.trendyol.com/kadin-sweatshirt-x-g1-c1179",
        "https://www.trendyol.com/kadin-elbise-x-g1-c56",
        "https://www.trendyol.com/kadin-jean-x-g1-c120",
        "https://www.trendyol.com/kadin-tayt-x-g1-c121"
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    all_products = []
    max_products = 5000  
    max_products_per_url = max_products // len(base_urls) 
    

    for base_url in base_urls:
        print(f"\nVeri çekme işlemi başlıyor: {base_url}")
        url_products = veri_cek(base_url, max_products_per_url, headers)
        all_products.extend(url_products)
        print(f"{base_url} için çekilen ürün sayısı: {len(url_products)}")
        time.sleep(4)


    df = pd.DataFrame(all_products, columns=["Ürün İsmi", "Fiyat", "Marka", "Ürün Kategori"])


    df['Fiyat'] = df['Fiyat'].str.replace(' TL', '', regex=True)  
    df['Fiyat'] = df['Fiyat'].str.replace(',', '.', regex=True) 
    df['Fiyat'] = pd.to_numeric(df['Fiyat'], errors='coerce')  
    

    df['Fiyat'].fillna(df['Fiyat'].mean(), inplace=True)
    

    df['Marka'] = df['Marka'].fillna("Bilinmiyor").astype('category').cat.codes
    df['Ürün Kategori'] = df['Ürün Kategori'].fillna("Bilinmiyor").astype('category').cat.codes
    
    
    df['Ürün İsmi'] = df['Ürün İsmi'].apply(lambda x: None if pd.isna(x) or x.strip() == "" else x)
    df = df.dropna(subset=['Ürün İsmi'])
    
    
    X = df[['Fiyat', 'Marka']]  
    y = df['Ürün Kategori']  
    

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    kf = KFold(n_splits=10, shuffle=True, random_state=42)


    rf_accuracies = []
    knn_accuracies = []
    dt_accuracies = []
    
    rf_confusion_matrices = []
    knn_confusion_matrices = []
    dt_confusion_matrices = []
    
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    rf_class_report = {}
    knn_class_report = {}
    dt_class_report = {}
    
    rf_results = []
    knn_results = []
    dt_results = []

    
    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_accuracies.append(accuracy_score(y_test, rf_pred))
        rf_confusion_matrices.append(confusion_matrix(y_test, rf_pred))
        # performanss
        rf_class_report = classification_report(y_test, rf_pred, output_dict=True)
        rf_class_report_df = pd.DataFrame(rf_class_report).transpose()
        
        rf_results.append({
            'Accuracy': accuracy_score(y_test, rf_pred),
            'Precision': precision_score(y_test, rf_pred, average='weighted'),
            'Recall': recall_score(y_test, rf_pred, average='weighted'),
            'F1-Score': f1_score(y_test, rf_pred, average='weighted')
        })

        
        knn = KNeighborsClassifier(n_neighbors=5)  # k=5 varsayılan
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_accuracies.append(accuracy_score(y_test, knn_pred))
        knn_confusion_matrices.append(confusion_matrix(y_test, knn_pred))
        # prfeormasn
        knn_class_report = classification_report(y_test, knn_pred, output_dict=True)
        knn_class_report_df = pd.DataFrame(knn_class_report).transpose()
        
        knn_results.append({
            'Accuracy': accuracy_score(y_test, knn_pred),
            'Precision': precision_score(y_test, knn_pred, average='weighted'),
            'Recall': recall_score(y_test, knn_pred, average='weighted'),
            'F1-Score': f1_score(y_test, knn_pred, average='weighted')
        })

        
    

        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        dt_pred = dt.predict(X_test)
        dt_accuracies.append(accuracy_score(y_test, dt_pred))
        dt_confusion_matrices.append(confusion_matrix(y_test, dt_pred))
        # performans
        dt_class_report = classification_report(y_test, dt_pred, output_dict=True)
        dt_class_report_df = pd.DataFrame(dt_class_report).transpose()
        
        dt_results.append({
            'Accuracy': accuracy_score(y_test, dt_pred),
            'Precision': precision_score(y_test, dt_pred, average='weighted'),
            'Recall': recall_score(y_test, dt_pred, average='weighted'),
            'F1-Score': f1_score(y_test, dt_pred, average='weighted')
        })


        y_pred = rf.predict(X_test)
        

        accuracies.append(accuracy_score(y_test, y_pred) * 100)

        precisions.append(precision_score(y_test, y_pred, average='weighted') * 100)

        recalls.append(recall_score(y_test, y_pred, average='weighted') * 100)
        
        f1_scores.append(f1_score(y_test, y_pred, average='weighted') * 100)



rf_results = []
knn_results = []
dt_results = []


for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_results.append({
        'Accuracy': accuracy_score(y_test, rf_pred),
        'Precision': precision_score(y_test, rf_pred, average='weighted'),
        'Recall': recall_score(y_test, rf_pred, average='weighted'),
        'F1-Score': f1_score(y_test, rf_pred, average='weighted')
    })
    

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_results.append({
        'Accuracy': accuracy_score(y_test, knn_pred),
        'Precision': precision_score(y_test, knn_pred, average='weighted'),
        'Recall': recall_score(y_test, knn_pred, average='weighted'),
        'F1-Score': f1_score(y_test, knn_pred, average='weighted')
    })
    

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_results.append({
        'Accuracy': accuracy_score(y_test, dt_pred),
        'Precision': precision_score(y_test, dt_pred, average='weighted'),
        'Recall': recall_score(y_test, dt_pred, average='weighted'),
        'F1-Score': f1_score(y_test, dt_pred, average='weighted')
    })


rf_df = pd.DataFrame(rf_results).mean().to_frame().T
rf_df.insert(0, 'Model', 'Random Forest')

knn_df = pd.DataFrame(knn_results).mean().to_frame().T
knn_df.insert(0, 'Model', 'kNN')

dt_df = pd.DataFrame(dt_results).mean().to_frame().T
dt_df.insert(0, 'Model', 'Decision Tree')


all_results_df = pd.concat([rf_df, knn_df, dt_df], ignore_index=True)


with pd.ExcelWriter('model_performance_results.xlsx') as writer:
    rf_df.to_excel(writer, sheet_name='Random Forest', index=False)
    knn_df.to_excel(writer, sheet_name='kNN', index=False)
    dt_df.to_excel(writer, sheet_name='Decision Tree', index=False)
    all_results_df.to_excel(writer, sheet_name='All Models Summary', index=False)

print("Her algoritma için performans değerleri excel dosyasına kaydedildi.")

    with pd.ExcelWriter('model_performance_results.xlsx') as writer:
        rf_class_report_df.to_excel(writer, sheet_name='Random Forest Performance', index=True)
        knn_class_report_df.to_excel(writer, sheet_name='knn Performance', index=True)
        dt_class_report_df.to_excel(writer, sheet_name='Decision Tree Performance', index=True)
    
    print("Performans sonuçları başarıyla kaydedildi.")

    average_accuracy = sum(accuracies) / len(accuracies)
    average_precision = sum(precisions) / len(precisions)
    average_recall = sum(recalls) / len(recalls)
    average_f1_score = sum(f1_scores) / len(f1_scores)
    

    print(f"Ortalama Accuracy: {average_accuracy:.2f}%")
    print(f"Ortalama Precision: {average_precision:.2f}%")
    print(f"Ortalama Recall: {average_recall:.2f}%")
    print(f"Ortalama F1-Skor: {average_f1_score:.2f}%")


    results_df = pd.DataFrame({
        'Model': ['Random Forest', 'kNN', 'Decision Tree'],
        'Doğruluk (%)': [np.mean(rf_accuracies)*100, np.mean(knn_accuracies)*100, np.mean(dt_accuracies)*100],
    })

    print(results_df)
    
    

with pd.ExcelWriter('enenenenengüncel.xlsx', mode='a', engine='openpyxl') as writer:
    results_df = pd.DataFrame({
        'Model': ['Random Forest', 'kNN', 'Decision Tree'],
        'Doğruluk (%)': [np.mean(rf_accuracies)*100, np.mean(knn_accuracies)*100, np.mean(dt_accuracies)*100],
    })
    

    with pd.ExcelWriter('enenenenengüncel.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        results_df.to_excel(writer, sheet_name='Model Sonuçları', index=False)
        
        
        rf_cm_df = pd.DataFrame(rf_confusion_matrices[0])  
        knn_cm_df = pd.DataFrame(knn_confusion_matrices[0])
        dt_cm_df = pd.DataFrame(dt_confusion_matrices[0])  
        
        rf_cm_df.to_excel(writer, sheet_name='RF Confusion Matrix', index=False)
        knn_cm_df.to_excel(writer, sheet_name='kNN Confusion Matrix', index=False)
        dt_cm_df.to_excel(writer, sheet_name='DT Confusion Matrix', index=False)
        

        rf_class_report_df = pd.DataFrame(rf_class_report).transpose()
        knn_class_report_df = pd.DataFrame(knn_class_report).transpose()
        dt_class_report_df = pd.DataFrame(dt_class_report).transpose()
    
        rf_class_report_df.to_excel(writer, sheet_name='RF Classification Report', index=True)
        knn_class_report_df.to_excel(writer, sheet_name='kNN Classification Report', index=True)
        dt_class_report_df.to_excel(writer, sheet_name='DT Classification Report', index=True)


    metrics_df = pd.DataFrame({
        'Accuracy': [average_accuracy],
        'Precision': [average_precision],
        'Recall': [average_recall],
        'F1-Score': [average_f1_score]
    })
    metrics_df.to_excel(writer, sheet_name='Average Metrics', index=False)

print("Çıktılar mevcut excel dosyasına kaydedildi: 'trendyol_verisi.xlsx'")
    
    
    
    
    results_df.to_excel(writer, sheet_name='Model Sonuçları', index=False)
    
    
    rf_cm_df = pd.DataFrame(rf_confusion_matrices[0])  
    knn_cm_df = pd.DataFrame(knn_confusion_matrices[0])
    dt_cm_df = pd.DataFrame(dt_confusion_matrices[0])  
    
    rf_cm_df.to_excel(writer, sheet_name='RF Confusion Matrix', index=False)
    knn_cm_df.to_excel(writer, sheet_name='kNN Confusion Matrix', index=False)
    dt_cm_df.to_excel(writer, sheet_name='DT Confusion Matrix', index=False)
    

    metrics_df = pd.DataFrame({
        'Accuracy': [average_accuracy],
        'Precision': [average_precision],
        'Recall': [average_recall],
        'F1-Score': [average_f1_score]
    })
    metrics_df.to_excel(writer, sheet_name='Average Metrics', index=False)

print("Çıktılar mevcut excel dosyasına kaydedildi: 'trendyol_verisi.xlsx'")


rf_pred = rf.predict(X_test)
rf_class_report = classification_report(y_test, rf_pred, output_dict=True)


with pd.ExcelWriter('samsunrize.xlsx') as writer:
    df.to_excel(writer, sheet_name='Ürün Verisi', index=False)


    for class_label, metrics in rf_class_report.items():
        if class_label != 'accuracy': 
            class_df = pd.DataFrame(metrics, index=[class_label])
            
            class_df.to_excel(writer, sheet_name=f'sınıf {class_label} performansı', index=True)