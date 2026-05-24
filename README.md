# XAI Pipeline — Explainable AI for ICU Mortality Prediction

Bitirme projesi: WiDS Datathon 2020 veri seti üzerinde yüksek doğrulukta tahmin üreten
ve bu tahminlerin **neden** yapıldığını uzman seviyesinde açıklayan 5 katmanlı bir XAI sistemi.

---

## Mimari

```
Katman 1 — Black-Box ML Model
    ↓ prediction + internal state
Katman 2 — SHAP Evidence Extraction
    ↓ ham SHAP değerleri
Katman 3 — Structured Evidence Construction
    ↓ pattern-based reasoning yapıları
Katman 4 — LLM Agentic Reasoning
    ↓ expert-oriented narrative explanation
Katman 5 — Explanation Quality Evaluation
    ↓ faithfulness / consistency / hallucination skorları
```

---

## Dataset

**WiDS Datathon 2020** — ICU hasta mortalite tahmini  
- 91.713 hasta kaydı  
- 185 feature (demografik, klinik, lab değerleri)  
- Hedef: `hospital_death` (binary classification)

Veri Kaggle'dan indirilmeli ve `data/raw/` altına konulmalıdır.  
Ham veri git'e dahil edilmez (bkz. `.gitignore`).

---

## Proje Yapısı

```
xai-project/
├── data/
│   ├── raw/          # Ham veri (git'e dahil edilmez)
│   └── processed/    # Temizlenmiş / dönüştürülmüş veri
├── notebooks/        # Keşif ve analiz notebook'ları
│   └── 01_eda.ipynb
├── src/              # Python modülleri (her katman için)
├── outputs/
│   ├── figures/      # Grafikler
│   └── reports/      # Raporlar
├── requirements.txt
└── README.md
```

---

## Kurulum

```bash
# Sanal ortam oluştur
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt

# Notebook'ları çalıştır
jupyter lab
```

---

## Katmanlar — Geliştirme Durumu

| Katman | Açıklama | Durum |
|--------|----------|-------|
| 0 | Exploratory Data Analysis | 🔄 Devam ediyor |
| 1 | Black-Box ML Model | ⏳ Planlanıyor |
| 2 | SHAP Evidence Extraction | ⏳ Planlanıyor |
| 3 | Structured Evidence Construction | ⏳ Planlanıyor |
| 4 | LLM Agentic Reasoning | ⏳ Planlanıyor |
| 5 | Explanation Quality Evaluation | ⏳ Planlanıyor |
