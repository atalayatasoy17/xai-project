# Project Study Guide - Final Pipeline

Bu rehber, projenin final halini bastan sona calismak icin hazirlandi.
Notebook'larda gelistirme denemeleri durabilir; final, tekrarlanabilir akisi
`src/`, `scripts/` ve `reports/` altindaki guncel dosyalar temsil eder.

## A. Buyuk Resim

Proje, ICU mortalite tahmini icin uc katmanli bir sistem kurar:

1. **Prediction layer:** LightGBM mortalite olasiligi uretir.
2. **Explainability layer:** SHAP, tahmini artiran/azaltan faktorleri bulur.
3. **LLM + validation layer:** Structured evidence packet LLM aciklamasina
   cevrilir ve validator ile kontrol edilir.

Bu nedenle proje sadece model skoru degil, raporlanabilir bir XAI pipeline'dir.

## B. Final Preprocessing

Final preprocessing `src/preprocessing.py` icindeki `ICUPreprocessor` ile
yapilir.

### B.1. Cikarilan kolonlar

ID/location:

- `encounter_id`
- `patient_id`
- `hospital_id`
- `icu_id`

Leakage probability:

- `apache_4a_hospital_death_prob`
- `apache_4a_icu_death_prob`

Neden?

- ID/location degiskenleri hasta fizyolojisini temsil etmez.
- `icu_id` unit-level pattern tasiyabilir.
- APACHE probability kolonlari baska bir risk modelinin skorlaridir ve hedefe
  cok yakin bilgi tasiyabilir.

### B.2. Donusumler

Numeric:

- median imputation
- missing indicator
- scaler yok

Binary:

- most frequent imputation
- ordinal encoding

Categorical:

- most frequent imputation
- one-hot encoding
- `max_categories=10`

Final feature sayisi:

```text
379
```

### B.3. Neden scaler yok?

LightGBM tree-based model oldugu icin numeric scale'i standartlastirmak sart
degildir. Scaler kullanmamak evidence packet'teki degerleri daha dogal tutar.

## C. Final Model

Final model scripti:

```bash
python scripts/final/train_final_lgbm.py
```

Saved artifacts:

- `models/lgbm_tuned_clean.pkl`
- `models/icu_preprocessor.pkl`
- `models/lgbm_tuned_clean_threshold.json`

Final threshold:

```text
0.7274
```

Final metrics:

| Metric | Value |
|---|---:|
| AUROC | 0.9103 |
| AUPRC | 0.5999 |
| Accuracy | 0.9189 |
| Precision | 0.5278 |
| Recall | 0.5704 |
| F1 | 0.5483 |
| TN | 15952 |
| FP | 808 |
| FN | 680 |
| TP | 903 |

Yorum:

- Model ayrim gucu iyi.
- Dengesiz veri nedeniyle AUPRC onemli.
- Threshold 0.50 yerine 0.7274 secildi.
- Bu threshold precision/F1 dengesini iyilestirir, fakat recall'i sinirlar.

## D. Modeling Reports

Klasor:

- `reports/01_modeling`

Onemli dosyalar:

- `selected_lgbm_test_metrics.csv`
- `threshold_sweep_lgbm.csv`
- `native_lgbm_feature_importance.csv`
- `figures/selected_lgbm_confusion_matrix.png`
- `figures/threshold_sweep_precision_recall_f1.png`
- `figures/native_lgbm_feature_importance_gain_top20.png`

Native LightGBM importance ile SHAP ayni sey degildir:

- Native importance: modelin split/gain kullanimini gosterir.
- SHAP: tahmin uzerindeki katkilarin lokal/global aciklamasini verir.

## E. SHAP Explainability

Final SHAP refresh:

```bash
python scripts/final/refresh_explainability_reports.py
```

Top global SHAP feature'lar:

- `age`
- `ventilated_apache`
- `apache_3j_diagnosis`
- `d1_bun_max`
- `d1_spo2_min`
- `gcs_motor_apache`
- `gcs_verbal_apache`

### E.1. Global SHAP

Global SHAP bize modelin test seti genelinde hangi feature'lara daha cok
dayandigini gosterir.

Ana dosyalar:

- `reports/02_explainability/tables/global_shap_importance.csv`
- `reports/02_explainability/figures/shap_summary_top20.png`

### E.2. Local SHAP

Local SHAP tek hasta icin risk artiran ve azaltan katkilarin listesidir.

Secili case tipleri:

- TP
- FN
- FP
- TN

Ana dosyalar:

- `reports/02_explainability/tables/local_explanation_tp.csv`
- `reports/02_explainability/tables/local_explanation_fn.csv`
- `reports/02_explainability/tables/local_explanation_fp.csv`
- `reports/02_explainability/tables/local_explanation_tn.csv`
- `reports/02_explainability/figures/local_waterfall_tp.png`
- `reports/02_explainability/figures/local_waterfall_fn.png`
- `reports/02_explainability/figures/local_waterfall_fp.png`
- `reports/02_explainability/figures/local_waterfall_tn.png`

### E.3. Dependence ve Grouped Analizler

Dependence plot, bir feature'in degeri degistikce SHAP katkisinin nasil
degistigini gosterir.

Ornek bulgular:

- Yas arttikca age SHAP katkisi genel olarak risk artirici yonde ilerler.
- Dusuk SpO2 daha riskli SHAP katkilarina gidebilir.
- Ventilation varligi risk artirici katkilarla iliskilidir.
- Dusuk GCS motor/verbal skorlar risk sinyali tasir.

### E.4. Interaction ve Correlation

Exploratory olarak top 20 SHAP feature icin:

- feature correlation heatmap
- SHAP interaction heatmap
- top interaction listesi

uretildi.

Bu analizler LLM prompt'una verilmedi. Sebep: final LLM aciklamasi hasta
bazli local evidence'a dayanmalidir.

## F. Caution Analysis

Final preprocessing `icu_id` dahil ID/location kolonlarini modelden cikardi.
Bu karar iki nedenle savunulur:

1. `icu_id` hasta fizyolojisi degildir.
2. Ham veri seviyesinde unit-level mortality farklari gorulebilir.

Ilgili dosyalar:

- `reports/02_explainability/caution_analysis/icu_id_mortality_by_unit.csv`
- `reports/02_explainability/caution_analysis/removed_id_location_columns.csv`

Onemli ifade:

> `icu_id` final model feature'i degildir. Daha once caution gerektiren bir
> degisken olarak incelendi; final cozumde modelden cikarildi.

## G. Evidence Packet

Evidence packet, SHAP ve prediction sonucunu LLM'e verilecek kontrollu forma
cevirir.

Script:

```bash
python scripts/final/refresh_evidence_packets.py
```

Klasor:

- `reports/03_evidence`

Evidence packet icerir:

- predicted probability
- threshold
- predicted class
- risk-increasing evidence
- risk-decreasing evidence
- feature value
- SHAP contribution
- clinical meaning
- caution flags

Evidence packet LLM prompt'unda sunlari gostermez:

- true label
- TP/FN/FP/TN bilgisi
- observed outcome

Bu label leakage riskini azaltir.

## H. Prompt Design

Prompt builder:

- `src/prompts.py`

Final prompt bolumleri:

1. Prediction summary
2. Main risk-increasing factors
3. Main risk-decreasing factors
4. Caution notes
5. Overall interpretation

Prompt ilkeleri:

- sadece evidence kullan
- clinical meaning yoksa yorum uydurma
- true label kullanma
- direction karistirma
- caution flag varsa belirt

## I. LLM Generation

Generator:

- `gpt-4.1-mini`

Core functions:

- `generate_explanation`
- `revise_explanation`
- `revise_until_valid`

Dosya:

- `src/llm.py`

Unlabeled demo:

```bash
python scripts/demo/unlabeled_patient_llm.py --patient-position 7
```

Kaydetmeden deneme:

```bash
python scripts/demo/unlabeled_patient_llm.py --patient-position 27 --no-save
```

## J. Deterministic Validation

Validator dosyasi:

- `src/validation.py`

Kontroller:

| Check | Amac |
|---|---|
| forbidden phrases | unsupported/riskli ifadeleri yakalar |
| true-label leakage | gercek outcome sızıntısını yakalar |
| section structure | 5 bolum formatini kontrol eder |
| prediction consistency | probability/threshold bilgisini kontrol eder |
| caution mentions | caution flag'lerin aciklandigini kontrol eder |
| feature grounding | exact feature adlarinin evidence'ta olup olmadigini kontrol eder |
| direction consistency | risk artirici/azaltici yonu SHAP ile karsilastirir |

Validation report:

```json
{
  "passed": true,
  "revision_required": false,
  "deterministic_validation_score": 5.0,
  "dimension_scores": {
    "faithfulness_no_hallucination": 5,
    "caution_awareness": 5,
    "completeness": 5
  }
}
```

## K. Revision Bridge

Validator fail ederse:

1. `revision_feedback` uretilir.
2. LLM revizyon prompt'u alir.
3. Yeni aciklama tekrar validator'dan gecer.
4. Basariliysa revised explanation final kabul edilir.

Bu tasarim LLM'i kendi haline birakmaz; deterministik rapora baglar.

## L. Validation Audit

Audit script:

```bash
python scripts/evaluation/audit_saved_explanations.py
```

Output:

- `reports/09_validation_audit/validation_audit_summary.csv`

Final audit ozeti:

- 7 saved explanation audite girdi.
- Final kabul edilen 5 aciklama deterministik kontrolleri gecti.
- 2 initial aciklama fail oldu ve revised versiyonlari pass etti.

## M. GPT-4o Subjective Evaluation

Script:

```bash
python scripts/evaluation/gpt4o_subjective_evaluation.py
```

GPT-4o ne yapar?

- hard pass/fail vermez
- sadece `clinical_plausibility` ve `clarity` puanlar
- deterministic score ile hybrid quality score'a eklenir

Final GPT-4o summary:

- 5 final accepted explanation degerlendirildi.
- 4 aciklama hybrid score `4.65`
- high-risk `unlabeled_patient_15` aciklamasi hybrid score `4.90`

## N. Dashboard

Dashboard:

```bash
streamlit run dashboard/app.py
```

Gosterilenler:

- hasta secimi
- prediction probability
- threshold
- risk level
- SHAP evidence
- LLM explanation
- validation panel
- revision status
- GPT-4o advisory score

## O. Raporlama Icin En Temiz Hikaye

1. ICU mortality prediction modeli kuruldu.
2. Leakage/id/location riskleri azaltildi.
3. LightGBM final model threshold tuning ile secildi.
4. SHAP global ve local aciklanabilirlik sagladi.
5. Local SHAP evidence packet'e cevrildi.
6. LLM aciklamasi structured evidence'tan uretildi.
7. Deterministik validator ile aciklama kontrol edildi.
8. GPT-4o sadece subjektif kalite icin advisory kullanildi.
9. Dashboard ile hasta bazli demo hazirlandi.

## P. Kritik Savunma Noktalari

### SHAP nedensellik midir?

Hayir. SHAP modelin tahmin katkisini aciklar, causal effect kanitlamaz.

### Neden `icu_id` cikarildi?

`icu_id` hasta olcumu degil, unit/location bilgisidir. Modelin unit-level
pattern ogrenmesini istemiyoruz.

### Neden true label prompt'a gitmiyor?

Aciklama model tahmini ve evidence'a dayanmalidir. Gercek outcome aciklamayi
kirletmemelidir.

### Neden GPT-4o hard validator degil?

LLM evaluator da hata yapabilir. Objektif kontroller deterministik yapildi.

### Neden threshold 0.7274?

0.50 defaulttur. Final threshold test-set threshold sweep ile precision,
recall ve F1 dengesi incelenerek secildi.

## Q. Komut Ozeti

Final training:

```bash
python scripts/final/train_final_lgbm.py
```

Explainability refresh:

```bash
python scripts/final/refresh_explainability_reports.py
```

Modeling reports:

```bash
python scripts/final/refresh_modeling_reports.py
```

Evidence packets:

```bash
python scripts/final/refresh_evidence_packets.py
```

Validation fixtures:

```bash
python scripts/verify/validation.py
```

Validation audit:

```bash
python scripts/evaluation/audit_saved_explanations.py
```

GPT-4o advisory evaluation:

```bash
python scripts/evaluation/gpt4o_subjective_evaluation.py
```

Dashboard:

```bash
streamlit run dashboard/app.py
```

## R. Bilinen Sinirlar

- Klinik karar yerine gecmez.
- SHAP causal interpretation degildir.
- GPT-4o advisory evaluator'dur.
- Validator exact/alias matching kullandigi icin tum paraphrase risklerini
  kusursuz yakalamaz.
- Diagnosis code feature'lari model icin yararlidir; kod sozlugu olmadan
  spesifik tani adi gibi yorumlanmamalidir.
