# Pipeline Demo Notes

## Amaç

Bu aşamada notebook'larda parça parça geliştirilen akış `.py` modüllerine taşındı. Amaç, ham bir hasta satırı geldiğinde aynı preprocessing kararlarını uygulayan, final modeli kullanan, tahmin üreten ve bu tahmini SHAP/evidence/prompt katmanına taşıyan tekrar kullanılabilir bir pipeline kurmaktır.

Bu bölümde `unlabeled.csv` henüz kullanılmadı. Önce pipeline'ın doğruluğunu bildiğimiz held-out test verisi üzerinde kanıtlamak istedik. Çünkü test setinde gerçek etiketler var ve daha önce notebook'ta üretilmiş `X_test.csv` ile karşılaştırma yapabiliyoruz.

## Kurulan Modüller

Pipeline kodu `src/` altında modüler şekilde kuruldu:

- `src/preprocessing.py`: ham WiDS formatındaki veriyi modelin beklediği processed feature formatına dönüştürür.
- `src/prediction.py`: final LightGBM modelini ve threshold bilgisini yükler, ölüm olasılığı ve sınıf tahmini üretir.
- `src/explainability.py`: tek hasta için local SHAP explanation üretir.
- `src/evidence.py`: SHAP tablosunu structured evidence packet formatına çevirir.
- `src/pipeline.py`: preprocessing, prediction, SHAP ve evidence adımlarını tek hasta için birleştirir.
- `src/prompts.py`: evidence packet'ten LLM'e verilecek açıklama prompt'unu üretir.

Bu yapı sayesinde pipeline şu akışı çalıştırabilir:

```text
raw patient
→ preprocessing
→ model prediction
→ local SHAP explanation
→ structured evidence packet
→ LLM prompt
```

## Preprocessing Doğrulaması

Önce preprocessing adımı doğrulandı. `training_v2.csv` yeniden yüklendi, aynı `train_test_split` ayarları kullanıldı ve `ICUPreprocessor` sadece training split üzerinde fit edildi.

Doğrulama script'i:

```text
scripts/01_verify_preprocessing.py
```

Kontrol edilenler:

- yeniden oluşturulan `X_test` shape'i
- kaydedilmiş `data/processed/X_test.csv` shape'i
- kolon sırası
- tüm değerlerin birebir eşleşmesi
- `y_test` eşleşmesi

Sonuç:

```text
Recreated X_test shape : (18343, 263)
Saved X_test shape     : (18343, 263)
Columns match          : True
Values match           : True
y_test matches         : True
```

Bu sonuç pipeline preprocessing kodunun notebook preprocessing çıktısını birebir yeniden ürettiğini gösterir.

## Prediction Doğrulaması

Preprocessing doğrulandıktan sonra final model prediction adımı test edildi.

Doğrulama script'i:

```text
scripts/02_verify_prediction.py
```

Bu script:

- ham `training_v2.csv` verisini yükledi
- aynı train/test split'i yaptı
- test split'i preprocessing pipeline ile dönüştürdü
- `models/lgbm_tuned_clean.pkl` modelini yükledi
- `models/lgbm_tuned_clean_threshold.json` içinden threshold = 0.50 değerini okudu
- final test metriklerini yeniden hesapladı

Sonuçlar notebook modelleme aşamasıyla birebir eşleşti:

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

Bu sonuç, `.py` pipeline'ın final model değerlendirmesini notebook dışından yeniden üretebildiğini gösterir.

## SHAP ve Evidence Doğrulaması

Prediction sonrası tek hasta için SHAP explanation üretildi.

Doğrulama script'leri:

```text
scripts/03_verify_explainability.py
scripts/04_verify_evidence.py
```

Seçilen örnek hasta:

```text
patient_label: test_patient_0
test_row_index: 72745
prediction_type: TN
y_true: 0
y_pred: 0
y_proba: 0.0128
```

Bu hasta için model düşük mortalite riski verdi ve gerçek etiket de 0 olduğu için prediction type `TN` oldu.

Evidence packet içinde iki ana kanıt grubu üretildi:

- `risk_increasing_evidence`: SHAP değeri pozitif olan ve mortalite riskini artıran faktörler
- `risk_decreasing_evidence`: SHAP değeri negatif olan ve mortalite riskini azaltan faktörler

Her evidence kaydı şu alanları içerir:

- feature
- observed value
- SHAP value
- direction
- clinical meaning
- caution flags

Örnek risk artırıcı faktörler:

- `d1_wbc_min = 22.0`
- `d1_resprate_max = 92.0`
- `apache_3j_diagnosis = 306.01`

Örnek risk azaltıcı faktörler:

- `age = 52.0`
- `ventilated_apache = 0.0`
- `d1_mbp_min = 89.0`
- `d1_spo2_min = 98.0`

Burada dikkat edilmesi gereken önemli nokta şudur: `clinical_meaning` genel klinik anlamı açıklar, SHAP yönü ise o hasta özelinde model katkısını gösterir. Örneğin `ventilated_apache` genel olarak ciddi hastalıkla ilişkilidir; fakat bu hastada değer 0 olduğu için model risk azaltıcı katkı vermiştir.

## Prompt Üretimi

Evidence packet'ten LLM'e verilecek prompt otomatik üretildi.

Doğrulama script'i:

```text
scripts/06_verify_prompt.py
```

Prompt içinde:

- hasta tahmini
- risk artırıcı evidence
- risk azaltıcı evidence
- klinik anlamlar
- caution flag bilgileri
- açıklama formatı
- hallucination ve true-label leakage önleyici kurallar

yer aldı.

Özellikle önceki LLM denemelerinde gözlenen riskleri azaltmak için prompt'a şu kurallar eklendi:

- sadece verilen evidence kullan
- klinik bilgi uydurma
- true label'ı açıklamada kullanma
- prediction correct/incorrect gibi ifadeler kullanma
- evidence içinde yoksa ölçü birimi veya normal range ekleme
- risk artırıcı ve azaltıcı evidence ayrımını doğru yap

Bu, LLM açıklamasının evidence-grounded kalması için önemlidir.

### Label Leakage Temizliği

Bu aşamada prompt construction tekrar kontrol edildi ve önemli bir metodolojik ayrım netleştirildi:

```text
evidence packet = bizim analiz kaydımız
LLM prompt      = modelin gördüğü bilgi
```

Evidence packet içinde `y_true` ve `prediction_type` bilgileri kalabilir; çünkü bunlar test hastalarında hata analizi, TP/FN/FP/TN sınıflaması ve raporlama için gereklidir. Ancak bu bilgiler LLM'e açıklama üretimi sırasında verilmemelidir. Aksi halde açıklama, modelin kendi prediction evidence'ı yerine gerçek sonuçtan veya `TP/TN/FN/FP` bilgisinden etkilenebilir.

Bu nedenle `src/prompts.py` güncellendi. Prompt artık şunları içermez:

- `True label`
- `Case type`
- `TP`, `FN`, `FP`, `TN` bilgisi

Promptta yalnızca modelin kendi çıktıları ve evidence kalır:

```text
Predicted label
Predicted mortality probability
Decision threshold
Risk-increasing SHAP evidence
Risk-decreasing SHAP evidence
Caution flags
```

Bu değişiklikten sonra `scripts/06_verify_prompt.py`, `scripts/07_run_test_patient_demo.py`, `scripts/10_run_saved_artifact_patient_demo.py`, `scripts/11_run_unlabeled_patient_demo.py`, `scripts/08_run_test_patient_llm_demo.py` ve `scripts/12_run_unlabeled_patient_llm_demo.py` tekrar çalıştırıldı. Böylece kaydedilmiş demo prompt ve LLM çıktıları temiz prompt yapısıyla güncellendi.

## Test Hasta Demo Çıktıları

Son olarak full test patient demo çalıştırıldı.

Script:

```text
scripts/07_run_test_patient_demo.py
```

Bu script held-out test setinden bir hasta seçip tüm pipeline'ı çalıştırdı ve çıktıları kaydetti:

- `reports/07_pipeline_demo/test_patient_0_prediction.json`
- `reports/07_pipeline_demo/test_patient_0_evidence.json`
- `reports/07_pipeline_demo/test_patient_0_prompt.txt`

Bu demo, pipeline'ın tek hasta için uçtan uca çalıştığını gösterir:

```text
raw test patient
→ preprocessing
→ prediction
→ SHAP explanation
→ evidence packet
→ LLM prompt
```

## LLM Generation ve Validation

Prompt üretimi doğrulandıktan sonra pipeline'a opsiyonel LLM generation adımı eklendi.

Script:

```text
scripts/08_run_test_patient_llm_demo.py
```

Bu script aynı test hastası için şu akışı çalıştırır:

```text
raw test patient
→ preprocessing
→ prediction
→ SHAP explanation
→ evidence packet
→ LLM prompt
→ OpenAI explanation generation
→ deterministic validation
→ revision if needed
→ revised validation
```

LLM generation için `gpt-4.1-mini` kullanıldı. API anahtarı `.env` dosyasındaki `OPENAI_API_KEY` üzerinden okunur. API key yoksa LLM adımı anlaşılır bir hata mesajı verir.

Bu adım sonunda şu dosyalar oluşturuldu:

- `reports/07_pipeline_demo/test_patient_0_llm_evidence.json`
- `reports/07_pipeline_demo/test_patient_0_llm_prompt.txt`
- `reports/07_pipeline_demo/test_patient_0_llm_explanation.txt`
- `reports/07_pipeline_demo/test_patient_0_llm_validation.json`

Generated explanation okunabilir ve genel akışı takip edebilir durumdadır; ancak LLM'in evidence dışına taşma riski tamamen ortadan kalkmamıştır. Bu nedenle açıklama üretildikten sonra deterministic validation uygulanmıştır.

Final validation katmanı yalnızca forbidden phrase taraması yapmaz. `src/validation.py` şu kontrolleri birlikte yürütür:

- unsupported / forbidden wording
- true-label leakage
- required section structure
- prediction probability consistency
- caution mentions
- feature grounding
- SHAP direction consistency

Bu test hasta çıktısında initial explanation validator tarafından fail edildi. Başlıca nedenler unsupported wording ve prediction probability consistency problemiydi:

```text
stable
adequate
```

Bu ifadeler, açıklamanın bazı yerlerinde evidence'tan daha yorumlayıcı bir dile kaydığını gösterir. Bu kötü bir sonuç olarak değil, pipeline'ın LLM çıktısını körlemesine kabul etmediğinin kanıtı olarak yorumlandı.

Bu nedenle LLM çıktısı şu şekilde ele alınmalıdır:

```text
generated draft explanation
→ deterministic validation report
→ revision if needed
→ revised validation report
→ final explanation candidate
```

Bu yaklaşım notebook 08'de kurulan agentic review mantığıyla uyumludur. LLM açıklaması üretilebilir, fakat klinik bağlamda final kabul edilmeden önce faithfulness, hallucination ve caution awareness açısından değerlendirilmelidir.

## LLM Revision Loop

Validation katmanı eklendikten sonra pipeline'a otomatik revision loop da eklendi. Amaç, LLM'in ilk açıklamasında flag'lenen ifadeler varsa bu açıklamayı doğrudan final kabul etmemek ve ikinci bir LLM çağrısıyla daha sıkı bir revizyon üretmektir.

Yeni akış şu şekildedir:

```text
initial LLM explanation
→ deterministic validation
→ if revision_required is true, build revision prompt from validation feedback
→ revised LLM explanation
→ revised validation
```

İlk LLM çıktısında validator unsupported wording ve probability consistency gibi sorunlar yakaladı. Revision prompt artık yalnızca flag'lenen kelimeleri değil, validation report içindeki yapılandırılmış feedback'i kullanır. Örneğin:

```text
Remove or rephrase unsupported/risky wording: stable, adequate.
Correct the predicted mortality probability to match the evidence.
```

Bu feedback, LLM'e açıklamayı evidence'a daha uygun şekilde yeniden yazmasını söyler.

Revision sonrası validation sonucu:

```text
Revised validation passed: True
Revised deterministic validation score: 5.0
```

Bu sonuç, otomatik revision loop'un ilk açıklamadaki sorunları düzeltebildiğini gösterir.

Ayrıca validator başlangıçta basit substring matching kullanıyordu. Bu yaklaşım `instability` gibi desteklenen klinik ifadelerin içinde geçen `stability` kelimesini yanlışlıkla flag'leyebilirdi. Bu nedenle validator whole-word matching kullanacak şekilde güncellendi. Daha sonra caution mention kontrolü de alias-aware hale getirildi. Böylece `icu_id` gibi teknik kolon adları yerine `ICU unit identifier` gibi klinik ifadeler, doğru caution diliyle birlikte kullanıldığında kabul edilebilir.

Bu adım projenin agentic kısmı için önemlidir. Pipeline artık yalnızca LLM explanation üretmekle kalmaz; aynı zamanda çıktıyı denetler, problemli ifade bulursa revizyon ister ve revize edilmiş açıklamayı tekrar kontrol eder.

## Saved Artifact Demo

İlk pipeline doğrulamaları sırasında `ICUPreprocessor` her script içinde yeniden train split üzerinde fit ediliyordu. Bu yöntem doğrulama için uygundur; çünkü notebook preprocessing çıktısını yeniden üretmeyi test eder. Ancak deployment benzeri kullanım için daha doğru yapı, preprocessing kurallarını da model gibi kaydetmektir.

Bu nedenle fitted preprocessor artifact olarak kaydedildi:

```text
models/icu_preprocessor.pkl
```

Bu artifact şu bilgileri içerir:

- high-missing kolon kararları
- missing indicator kolonları
- numeric ve categorical kolon listeleri
- train median değerleri
- final feature schema

Artifact kaydetme script'i:

```text
scripts/09_save_preprocessor_artifact.py
```

Bu script yalnızca preprocessor'ı kaydetmekle kalmadı, kaydedilen artifact'i tekrar yükleyip held-out test setini dönüştürdü ve kaydedilmiş `data/processed/X_test.csv` ile karşılaştırdı.

Doğrulama sonucu:

```text
Feature count: 263
X_test match : True
```

Bu, saved preprocessor artifact'inin notebook preprocessing çıktısını birebir yeniden üretebildiğini gösterir.

Ardından deployment-style demo script'i eklendi:

```text
scripts/10_run_saved_artifact_patient_demo.py
```

Bu script artık preprocessing'i yeniden fit etmez. Bunun yerine şu artifact'leri yükler:

```text
models/icu_preprocessor.pkl
models/lgbm_tuned_clean.pkl
models/lgbm_tuned_clean_threshold.json
```

Sonra raw test hastası üzerinde aynı pipeline'ı çalıştırır:

```text
saved preprocessor
+ saved model
+ saved threshold
+ raw patient
→ prediction
→ SHAP explanation
→ evidence packet
→ prompt
```

Saved artifact demo çıktıları:

- `reports/07_pipeline_demo/test_patient_0_saved_artifacts_prediction.json`
- `reports/07_pipeline_demo/test_patient_0_saved_artifacts_evidence.json`
- `reports/07_pipeline_demo/test_patient_0_saved_artifacts_prompt.txt`

Prediction sonucu önceki test patient demo ile aynı çıktı:

```text
death_probability: 0.0128
prediction: 0
threshold: 0.50
```

Bu adım `unlabeled.csv` demosu için doğrudan temel oluşturur. Çünkü etiketsiz yeni hasta verisinde artık training datasını yeniden fit etmeden, kaydedilmiş preprocessing ve model artifact'leri ile tahmin ve açıklama üretilebilir.

## Neden Unlabeled En Sona Bırakıldı?

`data/raw/unlabeled.csv` gerçek deployment demosu için uygundur; çünkü ham formatta gelir ve `hospital_death` değerleri boştur. Ancak etiketsiz olduğu için prediction doğruluğunu kontrol edemeyiz.

Bu nedenle önce held-out test verisi kullanıldı:

- gerçek etiketler biliniyor
- notebook çıktılarıyla birebir karşılaştırma yapılabiliyor
- preprocessing ve prediction doğrulanabiliyor
- SHAP/evidence/prompt akışı kontrollü şekilde incelenebiliyor

Bu doğrulama tamamlandıktan sonra aynı pipeline `unlabeled.csv` üzerinde güvenle uygulanabilir.

## Sonuç

Bu aşamada notebook tabanlı analizlerden tekrar kullanılabilir bir Python pipeline'a geçildi. Pipeline artık ham test hastasını alıp modelin beklediği feature formatına dönüştürebiliyor, final LightGBM modeliyle tahmin üretiyor, local SHAP explanation çıkarıyor, structured evidence packet oluşturuyor, LLM için evidence-grounded prompt hazırlıyor ve LLM açıklamasını deterministic validation/revision katmanından geçirebiliyor. Ayrıca saved preprocessor artifact ile deployment-style kullanım da doğrulandı.

Bu adım projenin deployment benzeri akışa geçişidir. Daha sonraki aşamalarda bu yapı genişletildi:

- aynı saved-artifact pipeline `unlabeled.csv` üzerinde çalıştırıldı
- saved LLM explanations deterministic audit'ten geçirildi
- alias-aware caution validation eklendi
- GPT-4o yalnızca clinical plausibility ve clarity için advisory evaluator olarak kullanıldı
