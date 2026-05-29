# Unlabeled Patient Demo Notes

## Amaç

Bu aşamada doğrulanmış saved artifact pipeline ilk kez `data/raw/unlabeled.csv` üzerinde çalıştırıldı. Amaç, proje pipeline'ının gerçek deployment senaryosuna benzer şekilde ham ve etiketsiz bir hasta satırını alıp tahmin, SHAP evidence ve LLM prompt çıktısı üretebildiğini göstermektir.

Bu demo bir performans değerlendirmesi değildir. `unlabeled.csv` içinde gerçek `hospital_death` etiketi olmadığı için accuracy, recall, precision veya confusion matrix hesaplanamaz.

## Kullanılan Akış

Bu demo artık training verisi üzerinde yeniden preprocessing fit etmez. Bunun yerine kaydedilmiş artifact'ler kullanılır:

```text
models/icu_preprocessor.pkl
models/lgbm_tuned_clean.pkl
models/lgbm_tuned_clean_threshold.json
```

Çalıştırılan script:

```text
scripts/run_unlabeled_patient_demo.py
```

Akış:

```text
raw unlabeled patient
→ saved preprocessor transform
→ saved LightGBM model prediction
→ threshold decision
→ local SHAP explanation
→ structured evidence packet
→ LLM prompt
```

## Üretilen Çıktılar

Demo çıktıları `reports/unlabeled_demo/` altında kaydedildi:

- `unlabeled_patient_0_prediction.json`
- `unlabeled_patient_0_evidence.json`
- `unlabeled_patient_0_prompt.txt`

Bu dosyalar sırasıyla model tahminini, SHAP tabanlı evidence packet'i ve LLM açıklaması için hazırlanmış prompt'u içerir.

## Prediction Sonucu

Seçilen hasta:

```text
patient_label: unlabeled_patient_0
original index: 0
```

Model tahmini:

```text
death_probability: 0.0291
prediction: 0
threshold: 0.50
```

Bu, modelin bu etiketsiz hasta için düşük mortalite olasılığı verdiğini ve threshold 0.50 altında kaldığı için sınıf tahminini `0` olarak ürettiğini gösterir.

Gerçek etiket olmadığı için bu tahminin doğru veya yanlış olduğu söylenemez. Bu demo yalnızca inference ve explanation pipeline'ın çalıştığını gösterir.

## Evidence Özeti

Evidence packet, model tahminini artıran ve azaltan SHAP katkılarını iki gruba ayırdı.

Başlıca risk artırıcı evidence:

- `gcs_verbal_apache = 1.0`
- `gcs_motor_apache = 5.0`
- `gcs_eyes_apache = 2.0`
- `d1_glucose_min = 167.0`
- `apache_3j_diagnosis = 405.01`
- `d1_sysbp_min = 79.0`
- `d1_bun_max = 19.0`
- `resprate_apache = 5.0`

Başlıca risk azaltıcı evidence:

- `icu_id = 1105`
- `age = 56.0`
- `ventilated_apache = 0.0`
- `h1_resprate_min = 8.0`
- `d1_resprate_max = 20.0`
- `d1_spo2_min = 96.0`
- `d1_wbc_min = 4.7`
- `d1_hemaglobin_min = 13.8`

## Caution Note

Evidence içinde `icu_id` risk azaltıcı yönde güçlü bir SHAP katkısı verdi:

```text
icu_id = 1105
shap_value = -0.8035
```

Ancak `icu_id` klinik bir hasta özelliği değil, ICU unit/location identifier niteliğinde bir değişkendir. Bu nedenle evidence packet otomatik caution flag ekledi:

```text
Non-clinical unit/location identifier; interpret cautiously.
```

Bu önemli bir noktadır. Model bu değişkenden sinyal almış olabilir, fakat bu sinyal hasta seviyesinde doğrudan klinik neden gibi yorumlanmamalıdır. Daha önce SHAP analizinde de `icu_id` için sensitivity analizi yapılması not edilmişti.

## LLM Prompt

`unlabeled_patient_0_prompt.txt` dosyası, evidence packet'i LLM açıklamasına dönüştürmek için hazırlandı.

Prompt içinde:

- prediction bilgisi
- risk-increasing evidence
- risk-decreasing evidence
- clinical meaning
- caution flags
- hallucination ve true-label leakage önleyici kurallar

yer alır.

Bu aşamada unlabeled hasta için LLM explanation henüz çalıştırılmadı. Önce prediction/evidence/prompt pipeline'ın etiketsiz veri üzerinde doğru çalıştığı gösterildi. İstenirse bir sonraki adımda test patient için kurulan LLM generation + validation + revision loop aynı unlabeled hasta üzerinde de çalıştırılabilir.

## Sonuç

Unlabeled demo başarıyla çalıştı. Pipeline ham etiketsiz hasta verisini kaydedilmiş preprocessing artifact ile model input formatına dönüştürdü, final LightGBM modeli ile ölüm olasılığı tahmini üretti, SHAP tabanlı evidence packet oluşturdu ve LLM açıklaması için prompt hazırladı.

Bu adım, projenin doğrulanmış test pipeline'ından deployment benzeri etiketsiz hasta inference senaryosuna geçtiğini gösterir.
