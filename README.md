# Explainable ICU Mortality Prediction Pipeline

Bu proje, WiDS Datathon 2020 ICU verisi üzerinde hastane içi mortalite riskini tahmin eden ve bu tahminleri SHAP, structured evidence ve LLM tabanlı açıklama katmanlarıyla yorumlayan uçtan uca bir XAI pipeline'ıdır.

Amaç yalnızca yüksek performanslı bir model kurmak değil, modelin bir hasta için neden belirli bir risk tahmini verdiğini izlenebilir, kanıta dayalı ve denetlenebilir şekilde açıklamaktır.

## Project Overview

Pipeline şu akışı izler:

```text
raw ICU patient data
→ preprocessing
→ LightGBM mortality prediction
→ local SHAP explanation
→ structured evidence packet
→ LLM explanation prompt
→ LLM explanation
→ validation and revision
```

Final deployment benzeri demo, kaydedilmiş artifact'lerle çalışır:

```text
models/icu_preprocessor.pkl
models/lgbm_tuned_clean.pkl
models/lgbm_tuned_clean_threshold.json
```

Bu sayede yeni veya etiketsiz bir hasta satırı geldiğinde training preprocessing yeniden fit edilmeden tahmin ve açıklama üretilebilir.

## Dataset

Kullanılan veri:

- WiDS Datathon 2020 ICU dataset
- Hedef değişken: `hospital_death`
- Problem tipi: binary classification
- Pozitif sınıf: hastane içi ölüm

Ham ve processed veri dosyaları Git'e dahil edilmez:

```text
data/raw/
data/processed/
```

## Methodology

### 1. EDA

İlk aşamada hedef dağılımı, eksik değer yapısı, klinik değişkenlerin dağılımları ve ölüm oranıyla ilişkili pattern'ler incelendi. Eksik değerlerin tamamen rastgele olmadığı görüldüğü için missingness bilgisi modelleme aşamasında ayrıca temsil edildi.

### 2. Preprocessing

Preprocessing kararları yalnızca train verisi üzerinden öğrenildi:

- ID kolonları kaldırıldı.
- Leakage riski taşıyan APACHE death probability kolonları kaldırıldı.
- Train setinde yüzde 50'den fazla eksik olan kolonlar drop edildi.
- Kalan eksik kolonlar için missingness indicator üretildi.
- Sayısal değişkenler train medyanı ile dolduruldu.
- Kategorik değişkenler `Unknown` ile dolduruldu ve one-hot encoded edildi.
- Train ve test kolonları aynı feature schema'ya hizalandı.

Bu adım sonradan `ICUPreprocessor` sınıfına taşındı ve artifact olarak kaydedildi:

```text
models/icu_preprocessor.pkl
```

### 3. Modeling

Denediğimiz ana modeller:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- balanced XGBoost / LightGBM
- Optuna tuned XGBoost / LightGBM

Pozitif sınıf az olduğu için model seçiminde yalnızca accuracy'ye bakılmadı. Özellikle AUPRC, recall, F1 ve confusion matrix birlikte değerlendirildi.

Kullanılan temel metrikler:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

AUPRC, dengesiz sınıf probleminde pozitif sınıfı yakalama kalitesini daha iyi gösterdiği için ana karşılaştırma metriklerinden biri olarak kullanıldı.

Final model:

```text
LightGBM Tuned Clean
threshold = 0.50
```

Final test sonuçları:

| Metric | Value |
| --- | ---: |
| AUROC | 0.9019 |
| AUPRC | 0.5824 |
| Accuracy | 0.9013 |
| Precision | 0.4486 |
| Recall | 0.6286 |
| F1 | 0.5235 |
| TN | 15537 |
| FP | 1223 |
| FN | 588 |
| TP | 995 |

Threshold 0.50 seçildi çünkü klinik bağlamda false negative sayısını azaltmak önemliydi.

### 4. SHAP Explainability

Final LightGBM modeli için SHAP analizi yapıldı:

- global feature importance
- local patient-level explanation
- TP, FN, FP, TN vaka analizleri
- error analysis

SHAP değeri şu şekilde yorumlandı:

```text
SHAP > 0  → prediction riskini artıran katkı
SHAP < 0  → prediction riskini azaltan katkı
```

Özellikle `icu_id`, sıfır vital sign değerleri ve negatif `pre_icu_los_days` gibi dikkat gerektiren değişkenler için caution note eklendi.

### 5. Structured Evidence

Ham SHAP tabloları LLM'e doğrudan verilmedi. Önce structured evidence packet formatına çevrildi:

```json
{
  "prediction": {
    "y_pred": 0,
    "y_proba": 0.0291,
    "threshold": 0.5
  },
  "risk_increasing_evidence": [],
  "risk_decreasing_evidence": []
}
```

Her evidence kaydı şu alanları içerir:

- feature
- observed value
- SHAP value
- direction
- clinical meaning
- caution flags

Bu yapı LLM açıklamasının evidence-grounded kalmasını sağlar.

### 6. LLM Explanation and Validation

LLM katmanında `gpt-4.1-mini` ile evidence packet'ten klinik açıklama üretildi. Ancak LLM çıktısı doğrudan final kabul edilmedi.

Eklenen kontrol katmanı:

```text
initial explanation
→ forbidden phrase validation
→ revision if needed
→ revised validation
```

Örneğin test patient ve unlabeled patient demolarında initial açıklamalar bazı fazla yorumlayıcı ifadeler içerdi. Validator bu ifadeleri yakaladı ve revision loop sonrası revised validation temiz çıktı:

```text
forbidden_phrases: []
```

Bu yaklaşım, LLM'in evidence dışına taşma riskine karşı hafif ama etkili bir agentic review mekanizması sağlar.

Label leakage'i önlemek için LLM prompt'u özellikle sınırlandırıldı. Test/evaluation amaçlı evidence packet içinde `y_true` ve `prediction_type` tutulabilir; ancak bu bilgiler LLM'e gönderilen prompt'a yazılmaz. LLM yalnızca modelin kendi çıktısını (`predicted label`, `predicted probability`, `threshold`) ve SHAP tabanlı evidence kayıtlarını görür. Böylece açıklama gerçek sonuçtan veya TP/FN/FP/TN bilgisinden etkilenmeden, model evidence'ına dayanır.

## Repository Structure

```text
xai-project/
├── data/
│   ├── raw/                    # Git'e dahil edilmez
│   └── processed/              # Git'e dahil edilmez
├── models/
│   ├── icu_preprocessor.pkl
│   ├── lgbm_tuned_clean.pkl
│   └── lgbm_tuned_clean_threshold.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model.ipynb
│   ├── 04_shap_explainability.ipynb
│   ├── 05_evidence_construction.ipynb
│   ├── 06_llm_reasoning.ipynb
│   ├── 07_explanation_evaluation.ipynb
│   └── 08_llm_generation_and_agentic_review.ipynb
├── reports/
│   ├── 01_modeling
│   ├── 02_explainability
│   ├── 03_evidence
│   ├── 04_llm_reasoning
│   ├── 05_evaluation
│   ├── 06_llm_generation
│   ├── 07_pipeline_demo
│   └── 08_unlabeled_demo
├── scripts/
│   ├── 01_verify_preprocessing.py
│   ├── 02_verify_prediction.py
│   ├── 03_verify_explainability.py
│   ├── 04_verify_evidence.py
│   ├── 05_verify_patient_pipeline.py
│   ├── 06_verify_prompt.py
│   ├── 07_run_test_patient_demo.py
│   ├── 08_run_test_patient_llm_demo.py
│   ├── 09_save_preprocessor_artifact.py
│   ├── 10_run_saved_artifact_patient_demo.py
│   ├── 11_run_unlabeled_patient_demo.py
│   └── 12_run_unlabeled_patient_llm_demo.py
└── src/
    ├── preprocessing.py
    ├── prediction.py
    ├── explainability.py
    ├── evidence.py
    ├── prompts.py
    ├── llm.py
    └── pipeline.py
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run core verification:

```bash
python scripts/01_verify_preprocessing.py
python scripts/02_verify_prediction.py
python scripts/03_verify_explainability.py
python scripts/04_verify_evidence.py
python scripts/05_verify_patient_pipeline.py
python scripts/06_verify_prompt.py
```

Run saved artifact demo:

```bash
python scripts/10_run_saved_artifact_patient_demo.py
```

Run unlabeled patient demo:

```bash
python scripts/11_run_unlabeled_patient_demo.py
```

Run LLM demos:

```bash
python scripts/08_run_test_patient_llm_demo.py
python scripts/12_run_unlabeled_patient_llm_demo.py
```

LLM demos require an `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

## Key Outputs

Model comparison:

```text
reports/01_modeling/final_model_comparison.csv
```

SHAP outputs:

```text
reports/02_explainability/
```

Evidence packets:

```text
reports/03_evidence/
```

Pipeline demos:

```text
reports/07_pipeline_demo/
reports/08_unlabeled_demo/
```

## Important Notes

- This project is for research and educational use.
- Model predictions are not clinical decisions.
- LLM explanations are generated drafts and should be interpreted with validation results.
- `icu_id` is treated cautiously because it may reflect unit-level patterns rather than patient-level clinical status.
- Raw data and API keys are intentionally excluded from Git.
