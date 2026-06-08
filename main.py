# 'src' klasörünün içindeki modüllerden çağırıyoruz
from src.data import prepare_data
from src.model import train_and_evaluate
from sklearn.metrics import accuracy_score, classification_report

def main():
    # 1. Veriyi hazırla ve böl
    print("Veri hazırlanıyor ve Train/Test olarak bölünüyor...")
    X_train, X_test, y_train, y_test = prepare_data()
    
    # 2. Modeli eğit ve çapraz doğrulama ile kontrol et
    model = train_and_evaluate(X_train, y_train)
    
    # 3. Test Seti ile Final Değerlendirmesi
    print("\n--- Final Test Seti Değerlendirmesi ---")
    print("Model daha önce HİÇ GÖRMEDİĞİ %20'lik test verisiyle sınanıyor...")
    
    tahminler = model.predict(X_test)
    test_basarisi = accuracy_score(y_test, tahminler)
    
    print(f"Test Seti Başarısı: % {test_basarisi * 100:.2f}")
    
    # Detaylı sınıflandırma raporu
    print("\nDetaylı Rapor (Classification Report):")
    print(classification_report(y_test, tahminler, target_names=['Sağlam (0)', 'Bozulacak (1)']))

if __name__ == "__main__":
    main()