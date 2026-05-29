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
scripts/verify_preprocessing.py
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
scripts/verify_prediction.py
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
scripts/verify_explainability.py
scripts/verify_evidence.py
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
scripts/verify_prompt.py
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

## Test Hasta Demo Çıktıları

Son olarak full test patient demo çalıştırıldı.

Script:

```text
scripts/run_test_patient_demo.py
```

Bu script held-out test setinden bir hasta seçip tüm pipeline'ı çalıştırdı ve çıktıları kaydetti:

- `reports/pipeline_demo/test_patient_0_prediction.json`
- `reports/pipeline_demo/test_patient_0_evidence.json`
- `reports/pipeline_demo/test_patient_0_prompt.txt`

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
scripts/run_test_patient_llm_demo.py
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
→ lightweight validation
```

LLM generation için `gpt-4.1-mini` kullanıldı. API anahtarı `.env` dosyasındaki `OPENAI_API_KEY` üzerinden okunur. API key yoksa LLM adımı anlaşılır bir hata mesajı verir.

Bu adım sonunda şu dosyalar oluşturuldu:

- `reports/pipeline_demo/test_patient_0_llm_evidence.json`
- `reports/pipeline_demo/test_patient_0_llm_prompt.txt`
- `reports/pipeline_demo/test_patient_0_llm_explanation.txt`
- `reports/pipeline_demo/test_patient_0_llm_validation.json`

Generated explanation okunabilir ve genel akışı takip edebilir durumdadır; ancak LLM'in evidence dışına taşma riski tamamen ortadan kalkmamıştır. Bu nedenle açıklama üretildikten sonra lightweight validation uygulanmıştır.

Validation katmanı açıklamada geçmesini istemediğimiz veya dikkat gerektiren ifadeleri tarar. Bu test hasta çıktısında validator şu ifadeleri işaretledi:

```text
stable
stability
adequate
```

Bu ifadeler, açıklamanın bazı yerlerinde evidence'tan daha yorumlayıcı bir dile kaydığını gösterir. Bu kötü bir sonuç olarak değil, pipeline'ın LLM çıktısını körlemesine kabul etmediğinin kanıtı olarak yorumlandı.

Bu nedenle LLM çıktısı şu şekilde ele alınmalıdır:

```text
generated draft explanation
→ validation check
→ review/revision if needed
→ final explanation
```

Bu yaklaşım notebook 08'de kurulan agentic review mantığıyla uyumludur. LLM açıklaması üretilebilir, fakat klinik bağlamda final kabul edilmeden önce faithfulness, hallucination ve caution awareness açısından değerlendirilmelidir.

## LLM Revision Loop

Validation katmanı eklendikten sonra pipeline'a otomatik revision loop da eklendi. Amaç, LLM'in ilk açıklamasında flag'lenen ifadeler varsa bu açıklamayı doğrudan final kabul etmemek ve ikinci bir LLM çağrısıyla daha sıkı bir revizyon üretmektir.

Yeni akış şu şekildedir:

```text
initial LLM explanation
→ forbidden phrase validation
→ if flagged phrases exist, build revision prompt
→ revised LLM explanation
→ revised validation
```

İlk LLM çıktısında validator şu ifadeleri yakaladı:

```text
stable
adequate
abnormal
```

Bu ifadeler açıklamanın bazı bölümlerinde evidence'tan daha yorumlayıcı dile kayabileceğini gösterdi. Bunun üzerine revision prompt oluşturuldu. Revision prompt, LLM'e flag'lenen ifadeleri kaldırmasını veya evidence'a daha uygun şekilde yeniden yazmasını söyledi.

Revision sonrası validation sonucu:

```text
Revised forbidden phrases: []
```

Bu sonuç, otomatik revision loop'un ilk açıklamadaki sorunlu ifadeleri azaltabildiğini gösterir.

Ayrıca validator başlangıçta basit substring matching kullanıyordu. Bu yaklaşım `instability` gibi desteklenen klinik ifadelerin içinde geçen `stability` kelimesini yanlışlıkla flag'leyebilirdi. Bu nedenle validator whole-word matching kullanacak şekilde güncellendi. Böylece `stability` tek başına geçtiğinde yakalanır, ancak `instability` gibi farklı bir kelimenin içinde geçtiğinde false positive üretilmez.

Bu adım projenin agentic kısmı için önemlidir. Pipeline artık yalnızca LLM explanation üretmekle kalmaz; aynı zamanda çıktıyı denetler, problemli ifade bulursa revizyon ister ve revize edilmiş açıklamayı tekrar kontrol eder.

## Neden Unlabeled En Sona Bırakıldı?

`data/raw/unlabeled.csv` gerçek deployment demosu için uygundur; çünkü ham formatta gelir ve `hospital_death` değerleri boştur. Ancak etiketsiz olduğu için prediction doğruluğunu kontrol edemeyiz.

Bu nedenle önce held-out test verisi kullanıldı:

- gerçek etiketler biliniyor
- notebook çıktılarıyla birebir karşılaştırma yapılabiliyor
- preprocessing ve prediction doğrulanabiliyor
- SHAP/evidence/prompt akışı kontrollü şekilde incelenebiliyor

Bu doğrulama tamamlandıktan sonra aynı pipeline `unlabeled.csv` üzerinde güvenle uygulanabilir.

## Sonuç

Bu aşamada notebook tabanlı analizlerden tekrar kullanılabilir bir Python pipeline'a geçildi. Pipeline artık ham test hastasını alıp modelin beklediği feature formatına dönüştürebiliyor, final LightGBM modeliyle tahmin üretiyor, local SHAP explanation çıkarıyor, structured evidence packet oluşturuyor ve LLM için evidence-grounded prompt hazırlıyor.

Bu adım projenin deployment benzeri akışa geçişidir. Sonraki doğal adımlar:

- LLM explanation çıktısını evaluator/revision katmanıyla daha sıkı denetlemek
- aynı pipeline'ı `unlabeled.csv` üzerinde en son demo olarak çalıştırmak
- pipeline kodunu README ve final raporda metodolojik akış olarak özetlemek
