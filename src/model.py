from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def train_and_evaluate(X_train, y_train):
    """Modeli eğitir ve Cross Validation ile test eder."""
    
    # [KAVRAM]: Model Tanımlama & Overfitting'i Engelleme (Regularization mantığı)
    # max_depth=5 diyerek ağacın çok fazla büyümesini ve veriyi ezberlemesini engelliyoruz.
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # [KAVRAM]: Cross Validation (Çapraz Doğrulama)
    # Eğitim setini kendi içinde 5'e bölüp modeli 5 kez test ediyoruz. 
    # Böylece modelin başarısının tesadüf olup olmadığını (Varyansını) anlıyoruz.
    cv_skorlari = cross_val_score(model, X_train, y_train, cv=5)
    
    print("\n--- Çapraz Doğrulama (Cross Validation) Sonuçları ---")
    print(f"5 Farklı Denemenin Skorları: {cv_skorlari}")
    print(f"Ortalama Eğitim Başarısı: % {cv_skorlari.mean() * 100:.2f}")
    
    # Modeli tüm eğitim verisiyle kalıcı olarak eğitiyoruz
    model.fit(X_train, y_train)
    
    return model