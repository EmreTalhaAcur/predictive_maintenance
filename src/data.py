import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    """Motor sensör verilerini simüle eder ve gerçek dünya 'Gürültüsü' ekler."""
    np.random.seed(42)
    
    # 1000 adet sensör verisi üretiyoruz
    sicaklik = np.random.normal(90, 10, 1000)
    titresim = np.random.normal(5, 2, 1000)
    saat = np.random.randint(100, 5000, 1000)
    
    # 1. KUSURSUZ KURAL (Eski deterministik sistem)
    kusursuz_ariza = ((sicaklik > 105) | (titresim > 8) | (saat > 4500)).astype(int)
    
    # 2. GÜRÜLTÜ (NOISE) EKLEME
    # Gerçek hayatta sağlam görünen motor bozulabilir veya sınırları zorlayan motor dayanabilir.
    # Verilerin rastgele %15'inin sonucunu tam tersine çeviriyoruz (1'leri 0, 0'ları 1 yapıyoruz)
    gurultu_maskesi = np.random.rand(1000) < 0.15 
    gercekci_ariza = np.where(gurultu_maskesi, 1 - kusursuz_ariza, kusursuz_ariza)
    
    # DataFrame oluşturuyoruz
    X = pd.DataFrame({'Sicaklik': sicaklik, 'Titresim': titresim, 'Calisma_Saati': saat})
    y = gercekci_ariza
    
    # Veriyi eğitim ve test olarak bölüyoruz
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test