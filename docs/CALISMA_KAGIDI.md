# XAI ICU Mortality Project - Calisma Kagidi

Bu kagit projenin final halini ogrenmek ve sunum/rapor hazirlamak icin
hazirlandi. Eski notebook denemeleri projede durabilir, fakat final
tekrarlanabilir akisi `src/`, `scripts/` ve `reports/` altindaki guncel
ciktilar temsil eder.

## 1. Projenin Amaci

Bu proje ICU hastalari icin hastane mortalite riskini tahmin eden bir
LightGBM modeli kurar ve bu tahmini aciklanabilir hale getirir.

Temel fikir:

1. Ham hasta verisi temizlenir ve modele uygun sayisal matrise donusturulur.
2. LightGBM modeli mortalite olasiligi uretir.
3. SHAP ile tahmini artiran ve azaltan faktorler bulunur.
4. Bu bilgiler structured evidence packet formatina cevrilir.
5. LLM bu paketten okunabilir klinik aciklama uretir.
6. Aciklama deterministik validator ile kontrol edilir.
7. GPT-4o sadece subjektif kalite boyutlari icin advisory evaluator olarak kullanilir.

Kisa anlatim:

> Model sadece tahmin uretmiyor; tahminin hangi kanitlara dayandigini,
> hangi ifadelerin guvenli oldugunu ve aciklamanin evidence'a sadik kalip
> kalmadigini da sistematik olarak raporluyoruz.

## 2. Veri ve Target

Ana veri:

- `data/raw/training_v2.csv`

Target:

- `hospital_death`

Target anlami:

- `0`: hasta hastanede olmemis
- `1`: hasta hastanede olmus

Unlabeled demo verisi:

- `data/raw/unlabeled.csv`

Submission iskeleti:

- `data/raw/sample_submission.csv`

Final pipeline unlabeled hastalarda gercek label kullanmaz. Bu, demo
aciklamalarinin gercek outcome'a baglanmasini engeller.

## 3. Final Preprocessing Karari

Final preprocessing `src/preprocessing.py` icindeki `ICUPreprocessor` ile
yapilir.

Final kararlar:

- ID/location kolonlari cikarildi:
  - `encounter_id`
  - `patient_id`
  - `hospital_id`
  - `icu_id`
- APACHE leakage probability kolonlari cikarildi:
  - `apache_4a_hospital_death_prob`
  - `apache_4a_icu_death_prob`
- Numeric kolonlar:
  - median imputation
  - missingness indicator
  - StandardScaler yok
- Binary/categorical kolonlar:
  - missing degerler doldurulur
  - binary icin ordinal encoding
  - categorical icin OneHotEncoder
  - nadir kategoriler `max_categories=10` ile kontrol edilir

Final feature sayisi:

- `379`

Neden StandardScaler yok?

LightGBM tree-based bir modeldir. Tree modellerinde feature scaling genelde
gerekli degildir. Scale kaldirilinca raw degerler, SHAP aciklamalari ve LLM
evidence packet daha okunabilir kalir.

## 4. Train/Test Split

Final egitim scripti:

- `scripts/16_train_final_lgbm_experiment.py`

Split mantigi:

- `train_test_split`
- `test_size=0.2`
- `random_state=42`
- `stratify=y`

Neden stratify?

Mortalite sinifi dengesiz oldugu icin train ve test setlerinde olum oraninin
benzer kalmasi gerekir.

Saved processed outputs:

- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `data/processed/feature_names.csv`

## 5. Final Model

Final model:

- Tuned LightGBM

Saved artifacts:

- `models/lgbm_tuned_clean.pkl`
- `models/icu_preprocessor.pkl`
- `models/lgbm_tuned_clean_threshold.json`

Final threshold:

```text
0.7274
```

Threshold neden 0.50 degil?

0.50 default bir esiktir, fakat dengesiz klinik sinif probleminde en iyi karar
esigi olmak zorunda degildir. Biz threshold sweep ile precision, recall ve F1
dengesini inceledik. Final threshold F1 odakli secildi.

Final test metrikleri:

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

- AUROC yuksek: model genel ayrim gucune sahip.
- AUPRC daha anlamli: olum sinifi nadir oldugu icin precision-recall daha
  kritik.
- Threshold 0.7274 ile precision/F1 dengesi iyilesti.
- Recall dusuk kalabilir; bu klinik risk olarak raporda belirtilmelidir.

## 6. Native LightGBM Feature Importance

Native LightGBM importance modelin split/gain bilgisini verir.

Saved table:

- `reports/01_modeling/native_lgbm_feature_importance.csv`

En onemli gain feature'lar:

- `ventilated_apache`
- `apache_3j_diagnosis`
- `gcs_motor_apache`
- `age`
- `d1_sysbp_min`
- `d1_lactate_min`
- `d1_spo2_min`

Bu black-box modelin hangi degiskenlerden faydalandigini gosterir, fakat
tek basina lokal hasta aciklamasi degildir. Lokal yorum icin SHAP kullanilir.

## 7. SHAP Analizi

SHAP, model tahminini feature katkilarina ayirir.

Kullandigimiz iki temel seviye:

- Global SHAP: model genel olarak hangi feature'lara dayaniyor?
- Local SHAP: tek hasta icin hangi feature riski artirdi/azaltti?

Final SHAP outputs:

- `reports/02_explainability/tables/global_shap_importance.csv`
- `reports/02_explainability/figures/shap_summary_top20.png`
- `reports/02_explainability/figures/local_waterfall_tp.png`
- `reports/02_explainability/figures/local_waterfall_fn.png`
- `reports/02_explainability/figures/local_waterfall_fp.png`
- `reports/02_explainability/figures/local_waterfall_tn.png`

Top global SHAP feature'lar:

- `age`
- `ventilated_apache`
- `apache_3j_diagnosis`
- `d1_bun_max`
- `d1_spo2_min`
- `gcs_motor_apache`
- `gcs_verbal_apache`

Onemli not:

SHAP korelasyon/katki gosterir; nedensellik kanitlamaz.

## 8. Dependence, Interaction ve Correlation

Exploratory SHAP analizinde sunlara da bakildi:

- top 20 SHAP feature
- dependence plots
- top 20 feature correlation heatmap
- top SHAP interaction pairs

Saved outputs:

- `reports/02_explainability/tables/top20_shap_features.csv`
- `reports/02_explainability/tables/top20_shap_interactions.csv`
- `reports/02_explainability/figures/top20_shap_interaction_heatmap.png`
- `reports/02_explainability/figures/top20_feature_correlation_heatmap.png`

Bu analizler LLM'e dogrudan verilmez. Neden?

LLM explanation local hasta evidence packet'e dayanmalidir. Interaction/global
analizler raporlama ve model anlama icin kullanilir; hasta bazli aciklamayi
kalabaliklastirmamak icin prompt'a eklenmez.

## 9. Caution Flag Karari

Caution flag, bir feature'in yorumlanirken dikkat gerektirdigini belirtir.

Final durumda `icu_id` modelden cikarildi. Bunun nedeni:

- `icu_id` hasta fizyolojisi degil, unit/location bilgisidir.
- Ham veri analizinde unit-level mortality farklari gorulebilir.
- Bu tur degiskenler site/unit pattern tasiyabilir.
- Bu nedenle final preprocessing icinde ID/location kolonlari tamamen cikarildi.

Hala caution gerektiren durumlara ornek:

- zero-valued vital signs
- negatif veya garip sure/timing degerleri
- dogrudan klinik olcum olmayan operasyonel degiskenler

## 10. Evidence Packet Nedir?

Evidence packet, model tahmininden LLM'e giden kontrollu ara formattir.

Icerir:

- prediction probability
- threshold
- predicted class
- risk-increasing SHAP evidence
- risk-decreasing SHAP evidence
- feature value
- clinical meaning
- caution flags

Icermez:

- true label
- TP/FN/FP/TN bilgisi
- gercek outcome

Neden yapiyoruz?

LLM'e ham dataframe veya tum SHAP tablosunu vermek yerine sadece gerekli,
denetlenebilir, structured evidence veriyoruz. Bu hallucination riskini azaltir.

## 11. Prompt Tasarimi

Prompt, evidence packet'i okunabilir bolumlere cevirir.

Beklenen 5 bolum:

1. Prediction summary
2. Main risk-increasing factors
3. Main risk-decreasing factors
4. Caution notes
5. Overall interpretation

Prompt kurali:

- sadece verilen evidence kullan
- true label kullanma
- direction karistirma
- clinical meaning yoksa yorum uydurma
- caution flag varsa belirt

## 12. LLM Generation

Generator:

- `gpt-4.1-mini`

Gorevi:

- evidence packet'ten klinik olarak okunabilir aciklama uretmek

Final demo scriptleri:

- `scripts/08_run_test_patient_llm_demo.py`
- `scripts/12_run_unlabeled_patient_llm_demo.py`

Kaydetmeden deneme:

```bash
python scripts/12_run_unlabeled_patient_llm_demo.py --patient-position 27 --no-save
```

## 13. Deterministic Validator

Validator:

- `src/validation.py`

Kontroller:

- forbidden phrases
- true-label leakage
- section structure
- prediction consistency
- caution mention
- feature grounding
- direction consistency

Validator neden gerekli?

LLM'ler okunabilir ama bazen evidence disi yorum yapabilir. Deterministik
validator gecis kapisidir; aciklama pass/fail kararini GPT-4o'ya birakmayiz.

Validation report:

- `passed`
- `revision_required`
- `deterministic_validation_score`
- `dimension_scores`
- `checks`
- `revision_feedback`

## 14. Revision Loop

Eger validator `revision_required=True` derse:

1. revision feedback LLM'e verilir
2. aciklama revize edilir
3. revize metin tekrar validator'dan gecer
4. en fazla belirlenen round kadar denenir

Bu bridge `src/llm.py` icindedir.

## 15. GPT-4o Evaluator

GPT-4o final pipeline'da hard validator degildir.

Sadece advisory olarak su iki subjektif boyutu puanlar:

- clinical plausibility
- clarity

Neden?

Faithfulness, leakage, probability, direction gibi objektif kontrolleri
deterministik yapmak daha guvenlidir. GPT-4o daha cok okunabilirlik ve klinik
makulluk gibi subjektif alanlarda raporlama destegi verir.

## 16. Dashboard

Dashboard:

- `dashboard/app.py`

Calistirma:

```bash
streamlit run dashboard/app.py
```

Dashboard'da gorulenler:

- hasta secimi
- prediction probability / threshold
- risk-increasing ve risk-decreasing evidence
- LLM explanation
- validation panel
- revision durumu
- GPT-4o advisory evaluation

## 17. En Onemli Savunma Cumleleri

- Bu proje sadece black-box prediction degil, evidence-grounded explanation pipeline'dir.
- SHAP lokal hasta seviyesinde model katkilarini verir; nedensellik iddiasi degildir.
- LLM aciklamalari kontrolsuz kabul edilmez.
- Final pipeline true label'i prompt'a gondermez.
- Deterministik validator hard gate olarak calisir.
- GPT-4o sadece subjective evaluation icin kullanilir.
- `icu_id` gibi location/id degiskenleri final modelden cikarildi.
- Final threshold 0.50 degil, test set threshold sweep ile secilen 0.7274'tur.

## 18. Hangi Dosya Ne Ise Yarar?

Core modules:

- `src/preprocessing.py`: final preprocessing
- `src/prediction.py`: model ve threshold yukleme/tahmin
- `src/explainability.py`: SHAP hesaplama
- `src/evidence.py`: evidence packet kurma
- `src/prompts.py`: LLM prompt olusturma
- `src/llm.py`: generation + revision
- `src/validation.py`: deterministic validation
- `src/pipeline.py`: tek hasta pipeline

Refresh scripts:

- `scripts/16_train_final_lgbm_experiment.py`
- `scripts/17_refresh_explainability_reports.py`
- `scripts/18_refresh_modeling_reports.py`
- `scripts/19_refresh_evidence_packets.py`

Verification/audit:

- `scripts/13_verify_validation.py`
- `scripts/14_audit_saved_explanations.py`
- `scripts/15_run_gpt4o_subjective_evaluation.py`

Final report folders:

- `reports/01_modeling`
- `reports/02_explainability`
- `reports/03_evidence`
- `reports/07_pipeline_demo`
- `reports/08_unlabeled_demo`
- `reports/09_validation_audit`
- `reports/10_gpt4o_evaluation`

## 19. Projenin Sinirlari

- Model klinik karar verici degildir; karar destegi/analiz amaclidir.
- SHAP nedensellik kanitlamaz.
- Categorical diagnosis codes model icin yararlidir ama klinik olarak kod sozlugu
  olmadan detayli tani ismi gibi yorumlanmamalidir.
- Validator exact/alias matching temellidir; tum paraphrase risklerini kusursuz
  yakalamaz.
- GPT-4o evaluator da hata yapabilir; bu nedenle advisory roldedir.
