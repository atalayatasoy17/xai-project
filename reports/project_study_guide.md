# XAI ICU Mortality Project Study Guide

Bu çalışma kağıdı, projenin baştan sona ne yaptığını, neden bu şekilde tasarlandığını ve kod yapısının nasıl çalıştığını öğrenmek için hazırlanmıştır. README projenin kısa ve profesyonel özetidir; bu dosya ise projenin arkasındaki kararları ve teknik akışı daha öğretici şekilde açıklar.

## 1. Projenin Temel Amacı

Bu projede ICU hastaları için hastane içi mortalite riski tahmini yapan bir makine öğrenmesi sistemi kuruldu. Ancak amaç sadece bir olasılık üretmek değildi. Amaç, modelin bu olasılığa neden ulaştığını açıklayabilen ve bu açıklamaları kontrol edebilen bir pipeline kurmaktı.

Temel problem şuydu:

```text
Ham hasta verisi → mortalite tahmini → SHAP açıklaması → LLM açıklaması → doğrulama
```

Burada kritik fikir:

```text
Açıklama, gerçek sonuçtan değil modelin kendi evidence'ından etkilenmelidir.
```

Yani hasta gerçekten ölmüş mü, yaşamış mı bilgisi LLM açıklamasına verilmemelidir. Açıklama sadece modelin gördüğü input, model tahmini ve SHAP evidence üzerinden üretilmelidir.

Bu nedenle proje üç ana hedef etrafında kuruldu:

- güçlü bir mortalite tahmin modeli kurmak
- model tahminlerini SHAP ile hasta seviyesinde açıklamak
- LLM açıklamalarını deterministik validator ve advisory GPT-4o evaluator ile denetlenebilir hale getirmek

## 2. Büyük Resim Pipeline

Final pipeline şu akışla çalışır:

```text
raw ICU patient data
→ fitted preprocessing artifact
→ LightGBM mortality prediction
→ local SHAP explanation
→ structured evidence packet
→ LLM explanation prompt
→ GPT-4.1-mini explanation
→ deterministic validation
→ revision if needed
→ validation audit
→ GPT-4o subjective evaluation
```

Bu akışta her katmanın görevi ayrıdır:

- Model tahmin yapar.
- SHAP modelin hangi feature'lardan nasıl etkilendiğini açıklar.
- Evidence packet SHAP bilgisini yapılandırılmış hale getirir.
- Prompt LLM'e sadece güvenli ve gerekli bilgileri verir.
- LLM klinik olarak okunabilir açıklama üretir.
- Validator açıklamayı evidence packet'e göre kontrol eder.
- Revision loop gerekiyorsa açıklamayı düzeltir.
- GPT-4o sadece subjektif kalite boyutlarını değerlendirir.

Bu ayrım projenin en önemli mimari kararlarından biridir. LLM'e hem açıklama üretme hem de hard validation yaptırmadık. Çünkü LLM judge halüsinasyon veya false positive üretebilir. Objektif kontrolleri deterministik yaptık.

## 3. Veri ve Problem Tanımı

Kullanılan veri WiDS Datathon 2020 ICU verisidir.

Hedef değişken:

```text
hospital_death
```

Problem tipi:

```text
binary classification
```

Sınıflar:

```text
0 → hasta hayatta kaldı
1 → hastane içi mortalite
```

Bu veri klinik olduğu için sınıf dengesizliği önemlidir. Pozitif sınıf, yani mortalite vakaları, negatif sınıfa göre daha azdır. Bu yüzden sadece accuracy'ye bakmak yeterli değildir.

Örneğin bir model çok fazla hastaya `0` derse accuracy yüksek görünebilir, ama mortalite vakalarını kaçırabilir. Bu nedenle model seçiminde şu metrikler birlikte değerlendirildi:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
AUROC
AUPRC
Confusion matrix
```

Klinik risk bağlamında recall önemlidir çünkü FN, yani gerçekten riskli hastayı düşük riskli görmek, kritik bir hata olabilir.

## 4. Preprocessing Aşaması

Preprocessing kararları notebook tarafında geliştirildi, sonra final reusable kod olarak `src/preprocessing.py` içine taşındı.

Ana sınıf:

```text
src/preprocessing.py → ICUPreprocessor
```

Bu sınıf ham WiDS verisini modelin beklediği feature formatına dönüştürür.

Yapılan temel işlemler:

- ID kolonları ve leakage riski taşıyan APACHE death probability kolonları çıkarıldı.
- Çok yüksek eksiklik oranı olan kolonlar düşürüldü.
- Eksik değerler için missingness indicator kolonları eklendi.
- Numeric kolonlar train medyanı ile dolduruldu.
- Categorical kolonlar `Unknown` ile dolduruldu.
- Categorical değişkenler one-hot encoding ile dönüştürüldü.
- Train ve test feature şemaları aynı sırada hizalandı.

Buradaki en önemli prensip:

```text
Preprocessing kararları sadece training split üzerinden öğrenilir.
```

Yani test verisinden medyan, kategori listesi veya kolon seçimi öğrenilmez. Bu, data leakage riskini azaltır.

Preprocessing doğrulaması:

```bash
python scripts/01_verify_preprocessing.py
```

Bu script yeniden oluşturulan `X_test` ile daha önce kaydedilmiş `X_test.csv` dosyasını karşılaştırır. Shape, kolon sırası ve değerlerin birebir eşleşmesi pipeline'ın notebook preprocessing sonucunu doğru yeniden ürettiğini gösterir.

## 5. Modelleme Aşaması

Model geliştirme notebook'larda yapıldı. Birden fazla model karşılaştırıldı:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- imbalance-weighted modeller
- Optuna tuned modeller

Final seçim:

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

Model ve threshold dosyaları:

```text
models/lgbm_tuned_clean.pkl
models/lgbm_tuned_clean_threshold.json
```

Prediction kodu:

```text
src/prediction.py
```

Bu dosya model yükleme, threshold yükleme ve hasta için prediction üretme işini yapar.

Prediction çıktısı temelde şunu verir:

```json
{
  "death_probability": 0.0291,
  "prediction": 0.0,
  "threshold": 0.5
}
```

Bu örnekte model mortalite olasılığını `0.0291` bulmuştur. Bu değer `0.50` threshold altında olduğu için sınıf tahmini `0` olmuştur.

## 6. SHAP Explainability Mantığı

Model tahmini tek başına yeterli değildir. Bu nedenle hasta seviyesinde modelin hangi feature'lardan nasıl etkilendiğini SHAP ile açıkladık.

Kod:

```text
src/explainability.py
```

SHAP mantığı:

```text
SHAP value > 0 → feature mortalite riskini artırdı
SHAP value < 0 → feature mortalite riskini azalttı
```

Burada dikkat edilmesi gereken nokta:

```text
Klinik anlam ile SHAP yönü aynı şey değildir.
```

Örneğin mekanik ventilasyon genel olarak ciddi hastalık göstergesidir. Ama `ventilated_apache = 0` ise bu hasta için ventilasyon yokluğu modelin riskini azaltabilir.

Yani:

```text
clinical meaning → feature'ın genel klinik yorumu
SHAP direction   → bu hasta için modelin tahminine katkı yönü
```

Bu ayrım LLM açıklamalarında çok önemlidir. LLM, "bu feature genel olarak kötü" bilgisini kullanıp SHAP yönünü ters anlatmamalıdır.

## 7. Evidence Packet Nedir?

SHAP çıktısı ham haliyle LLM'e verilmedi. Önce structured evidence packet formatına dönüştürüldü.

Kod:

```text
src/evidence.py
```

Evidence packet şunları içerir:

- hasta etiketi
- model prediction
- predicted probability
- threshold
- risk-increasing evidence
- risk-decreasing evidence
- feature value
- SHAP value
- clinical meaning
- caution flags

Örnek evidence yapısı:

```json
{
  "feature": "d1_spo2_min",
  "value": 59.0,
  "shap_value": 0.42,
  "direction": "risk_increasing",
  "clinical_meaning": "Low oxygen saturation may indicate hypoxemia.",
  "caution_flags": []
}
```

Evidence packet'in amacı LLM'e düzenli, sınırlı ve denetlenebilir bilgi vermektir.

Bu kararın nedeni:

```text
LLM ham veriden veya dağınık SHAP tablosundan serbest yorum üretirse hallucination riski artar.
```

Evidence packet sayesinde LLM sadece belirli feature'ları, belirli değerleri ve belirli klinik anlamları görür.

## 8. Caution Flag Mantığı

Her feature düz klinik neden gibi yorumlanmamalıdır. Bazı değişkenler model için sinyal taşısa bile hasta seviyesinde dikkatli yorumlanmalıdır.

Örnekler:

- `icu_id`: non-clinical unit/location identifier olabilir
- zero-valued vital signs: gerçek ekstrem durum veya kayıt hatası olabilir
- negatif `pre_icu_los_days`: veri kalitesi veya zamanlama problemi olabilir

Bu yüzden evidence packet içinde caution flag alanı vardır.

Örnek:

```text
icu_id = 1105
caution flag = Non-clinical unit/location identifier; interpret cautiously.
```

Bu şu anlama gelir:

```text
Model icu_id'den sinyal almış olabilir, fakat bunu hastanın klinik nedeni gibi yorumlamamalıyız.
```

Bu nokta rapor ve savunma için önemlidir. Çünkü XAI açıklamaları, modelin öğrendiği pattern'i gösterir; her pattern doğrudan klinik nedensellik anlamına gelmez.

## 9. Prompt Tasarımı

Prompt üretimi:

```text
src/prompts.py
```

Prompt, evidence packet'i LLM'in okuyabileceği bir açıklama görevine dönüştürür.

Prompt içinde şu bilgiler vardır:

- predicted label
- predicted mortality probability
- threshold
- risk-increasing evidence
- risk-decreasing evidence
- clinical meaning
- caution flags
- açıklama formatı
- hallucination önleyici kurallar

Prompt içinde özellikle bulunmaması gereken bilgiler:

```text
true label
case type
TP / FN / FP / TN bilgisi
prediction correct / incorrect bilgisi
```

Bu ayrım çok önemlidir:

```text
evidence packet = bizim analiz ve kayıt formatımız
LLM prompt      = LLM'in açıklama üretirken gördüğü bilgi
```

Evidence packet içinde `y_true` veya `prediction_type` tutulabilir, çünkü bunlar analiz ve raporlama için gereklidir. Ama LLM prompt'una gönderilmez.

Neden?

Çünkü projenin temel amacı şudur:

```text
Açıklama gerçek sonuçtan etkilenmesin.
```

Eğer prompt'a `True label = 1` veya `case_type = TP` girersek, LLM açıklamayı model evidence'ından değil gerçek sonuçtan etkilenerek yazabilir.

Bu nedenle `src/prompts.py` güncellendi ve label leakage riski temizlendi.

## 10. LLM Generation

LLM açıklama üretimi:

```text
src/llm.py
```

Kullanılan generation modeli:

```text
gpt-4.1-mini
```

Bu modelin görevi:

```text
structured evidence packet'ten kısa, okunabilir, evidence-grounded açıklama üretmek
```

Üretilen açıklama şu 5 bölümden oluşmalıdır:

```text
1. Prediction summary
2. Main risk-increasing factors
3. Main risk-decreasing factors
4. Caution notes
5. Overall interpretation
```

LLM'e verilen kurallar arasında şunlar vardır:

- sadece verilen evidence'ı kullan
- klinik bilgi uydurma
- true label'dan bahsetme
- correct/incorrect prediction deme
- evidence içinde yoksa ölçü birimi veya normal range ekleme
- risk-increasing ve risk-decreasing ayrımını doğru yap
- caution flag varsa belirt
- clinical_meaning yoksa ekstra yorum üretme

Bu kurallar tek başına yeterli değildir. Bu yüzden validator katmanı kuruldu.

## 11. Neden Validator Gerekliydi?

LLM açıklamaları akıcı ve ikna edici görünebilir, ama küçük hatalar içerebilir.

Örnek hatalar:

- evidence'ta olmayan yorum eklemek
- `stable`, `adequate`, `favorable` gibi desteklenmeyen ifadeler kullanmak
- predicted probability'yi yanlış aktarmak
- caution flag'i atlamak
- gerçek label'a referans vermek
- SHAP yönünü ters anlatmak

Bu hatalar objektif olarak kontrol edilebiliyorsa, bunları başka bir LLM'e sormak yerine deterministik kontrol etmek daha güvenlidir.

Bu nedenle:

```text
deterministik yapılabiliyorsa deterministik yap
subjektif kalan boyutları advisory LLM evaluator'a bırak
```

Bu proje kararının kısa özeti budur.

## 12. Deterministic Validator

Validator kodu:

```text
src/validation.py
```

Ana fonksiyon:

```text
validate_explanation(text, evidence_packet)
```

Bu fonksiyon açıklamayı evidence packet'e göre kontrol eder ve structured validation report döndürür.

Kontroller:

### 12.1 Forbidden Phrase Check

LLM'in fazla kesin veya evidence dışı yorumlayıcı ifadeler kullanıp kullanmadığını kontrol eder.

Örnek yasak/riskli ifadeler:

```text
stable
adequate
normal
abnormal
favorable
unfavorable
correct prediction
incorrect prediction
```

Örnek çıktı:

```json
{
  "passed": false,
  "found": ["favorable"]
}
```

### 12.2 True-Label Leakage Check

Açıklama gerçek outcome'dan bahsediyor mu kontrol eder.

Örnek problemli ifadeler:

```text
true label
true outcome
patient died
patient survived
correct prediction
incorrect prediction
```

Bu kontrol projenin temel amacını korur: açıklama gerçek sonuçtan etkilenmemelidir.

### 12.3 Section Structure Check

Açıklamada beklenen 5 bölüm var mı kontrol eder.

Beklenen bölümler:

```text
Prediction summary
Main risk-increasing factors
Main risk-decreasing factors
Caution notes
Overall interpretation
```

Eksik bölüm varsa explanation fail olur ve revision gerekir.

### 12.4 Prediction Consistency Check

LLM'in yazdığı probability evidence packet'teki probability ile uyumlu mu kontrol eder.

Örnek:

```text
evidence probability = 0.9919
LLM text = 0.99
```

Bu pass olur çünkü tolerans vardır:

```text
±0.01
```

Ama:

```text
evidence probability = 0.9919
LLM text = 0.199
```

Bu fail olur.

### 12.5 Caution Mention Check

Evidence packet içinde caution flag varsa, açıklamanın `Caution notes` bölümünde bu konu dikkatli şekilde belirtilmiş mi kontrol eder.

İlk versiyonda bu kontrol sadece exact feature name arıyordu. Örneğin `icu_id` literal olarak geçmezse fail diyordu. Fakat LLM doğru şekilde "ICU unit identifier" diyebiliyordu. Bu doğru bir klinik ifade olmasına rağmen fail oluyordu.

Bu yüzden caution check alias-aware hale getirildi.

Artık `icu_id` için şu ifadeler kabul edilebilir:

```text
icu_id
ICU unit identifier
unit identifier
unit-level
```

Ama iki koşul birlikte aranır:

```text
feature identity term + caution language
```

Yani sadece "unit identifier" demek yetmez; aynı cümlede dikkatli yorum dili de olmalıdır.

Örnek doğru caution:

```text
The ICU unit identifier should be interpreted cautiously because it is a non-clinical location variable.
```

Bu kontrolün amacı, validator'ın doğru doğal dili yanlışlıkla reddetmesini önlemektir.

### 12.6 Feature Grounding Check

Açıklamada exact feature adı geçiyorsa, bu feature evidence packet içinde var mı kontrol eder.

Örnek:

```text
Evidence packet: age, d1_spo2_min, ventilated_apache
Explanation: potassium_apache increased risk
```

Bu fail olur çünkü `potassium_apache` evidence packet içinde yoktur.

Sınırlama:

```text
Bu v1 kontrol exact feature-name matching kullanır.
```

Yani LLM `d1_spo2_min` yerine "oxygen saturation" derse bu kontrol onu tam değerlendiremez. Bu bilinçli olarak not edilmiştir.

### 12.7 Direction Consistency Check

Feature risk-increasing evidence içinde mi, risk-decreasing evidence içinde mi doğru anlatılmış mı kontrol eder.

Örnek:

```text
Evidence: d1_spo2_min → risk_increasing
Explanation: d1_spo2_min decreased risk
```

Bu sign flip hatasıdır.

Bu kontrol de v1'de exact feature-name üzerinden çalışır.

## 13. Validation Report Şeması

Validator tek bir structured report döndürür.

Örnek:

```json
{
  "passed": false,
  "revision_required": true,
  "deterministic_validation_score": 4.538,
  "dimension_scores": {
    "faithfulness_no_hallucination": 4,
    "caution_awareness": 5,
    "completeness": 5
  },
  "checks": {
    "forbidden_phrases": {
      "passed": false,
      "found": ["favorable"]
    }
  },
  "revision_feedback": [
    "Remove or rephrase unsupported/risky wording: favorable."
  ],
  "schema_version": "1.0"
}
```

Bu report iki nedenle önemlidir:

- Pipeline açıklamanın geçip geçmediğini makine tarafından okuyabilir.
- Audit script'i tüm açıklamaları aynı formatta analiz edebilir.

## 14. Deterministic Score Nasıl Hesaplanır?

Orijinal rubric boyutları:

```text
faithfulness_no_hallucination = 0.30
clinical_plausibility        = 0.25
caution_awareness            = 0.20
completeness                 = 0.15
clarity                      = 0.10
```

Deterministic validator sadece objektif kontrol edilebilen üç boyutu skorlar:

```text
faithfulness_no_hallucination
caution_awareness
completeness
```

Bu üç boyutun toplam ağırlığı:

```text
0.30 + 0.20 + 0.15 = 0.65
```

Bu yüzden skor normalize edilir:

```text
score =
(0.30 / 0.65) * faithfulness
+ (0.20 / 0.65) * caution_awareness
+ (0.15 / 0.65) * completeness
```

Sonuç 1-5 aralığında bir deterministic validation score verir.

Önemli:

```text
Bu skor GPT-4o evaluator skoru değildir.
```

Bu skor sadece deterministik olarak ölçülebilen kalite boyutlarını temsil eder.

## 15. Revision Loop

Validator fail verirse explanation doğrudan final kabul edilmez.

Akış:

```text
initial explanation
→ validate_explanation
→ revision_required = true
→ build revision feedback
→ revise_explanation
→ validate revised explanation
```

Kod:

```text
src/llm.py
```

Özellikle:

```text
revise_until_valid(...)
```

Bu fonksiyon açıklamayı validator report'a göre en fazla birkaç tur revize eder.

Revision prompt'a sadece "hata var" denmez; hangi hatalar olduğu structured feedback olarak verilir.

Örnek feedback:

```text
Remove or rephrase unsupported/risky wording: favorable.
Correct the predicted mortality probability to match the evidence.
Mention caution for flagged features: icu_id.
Fix direction for d1_spo2_min.
```

Bu sayede revision loop daha kontrollü çalışır.

## 16. Validation Fixture Tests

Validator'ın gerçekten beklenen hataları yakalayıp yakalamadığını test etmek için fixture script yazıldı.

Script:

```bash
python scripts/13_verify_validation.py
```

Bu script küçük yapay explanation örnekleriyle validator'ı test eder.

Fixture örnekleri:

- good explanation → pass
- ungrounded feature → fail
- sign flip → fail
- true label leak → fail
- missing section → fail
- wrong probability → fail
- missing caution → fail
- alias caution → pass

Bu testlerin amacı:

```text
Validator gerçek LLM çıktılarından önce kontrollü örneklerde doğru davranıyor mu?
```

Örneğin `alias_caution` fixture'ı, "ICU unit identifier" ifadesinin `icu_id` caution'ı için kabul edildiğini kanıtlar.

## 17. Validation Audit

Tek tek açıklamalara bakmak yeterli değildir. Kaydedilmiş tüm açıklamaları sistematik olarak kontrol etmek gerekir.

Script:

```bash
python scripts/14_audit_saved_explanations.py
```

Çıktı:

```text
reports/09_validation_audit/validation_audit_summary.csv
```

Bu CSV her explanation için şunları içerir:

- case_id
- source_file
- passed
- revision_required
- deterministic_validation_score
- forbidden_phrases
- true_label_leakage
- missing_sections
- prediction_consistency_passed
- missing_caution_mentions
- ungrounded_features
- direction_errors

Güncel audit sonucu:

```text
10 saved explanations audited
7 passed deterministic validation
3 failed due to unsupported wording
all revised explanations passed deterministic validation
```

Örnek fail nedenleri:

```text
test_patient_0 initial → stable, adequate
unlabeled_patient_0 initial → favorable
unlabeled_patient_7 initial → abnormal
```

Bu sonuç şunu gösterir:

```text
LLM açıklamaları çoğu zaman iyi, ama küçük unsupported wording hataları yapabiliyor.
Validator bunları yakalıyor ve revision loop düzeltebiliyor.
```

## 18. GPT-4o Evaluator Neden Ayrı?

Başta LLM judge ile açıklamaları değerlendirme fikri vardı. Fakat exploratory notebook aşamasında GPT-4o evaluator'ın false-positive faithfulness flag üretebildiği görüldü.

Bu bulgu önemliydi:

```text
LLM judge da hata yapabilir.
```

Bu yüzden final pipeline'da GPT-4o hard validator yapılmadı.

Final karar:

```text
Deterministic validator → hard gatekeeper
GPT-4o evaluator        → advisory subjective evaluator
```

GPT-4o sadece şu boyutları skorlar:

- clinical plausibility
- clarity

Script:

```bash
python scripts/15_run_gpt4o_subjective_evaluation.py
```

Kod:

```text
src/evaluator.py
```

GPT-4o evaluation sadece deterministic validation'dan geçmiş açıklamalara uygulanır.

Hybrid score şu mantıkla oluşur:

```text
deterministic scores:
- faithfulness_no_hallucination
- caution_awareness
- completeness

GPT-4o scores:
- clinical_plausibility
- clarity
```

Bu mimari daha savunulabilir:

```text
Objektif kontroller deterministik.
Subjektif kalite değerlendirmesi advisory LLM judge.
```

## 19. Test Patient ve Unlabeled Patient Demoları

İki ana demo tipi vardır.

### 19.1 Test Patient Demo

Script:

```bash
python scripts/08_run_test_patient_llm_demo.py
```

Bu demo held-out test setinden bir hasta seçer. Gerçek label vardır ama LLM prompt'una gönderilmez.

Amaç:

```text
Pipeline test verisinde doğru çalışıyor mu?
```

Kaydedilen çıktılar:

```text
reports/07_pipeline_demo/
```

Örnek:

```text
test_patient_0_llm_explanation.txt
test_patient_0_llm_validation.json
test_patient_0_llm_revised_explanation.txt
```

### 19.2 Unlabeled Patient Demo

Script:

```bash
python scripts/12_run_unlabeled_patient_llm_demo.py --patient-position 15 --no-save
```

Bu demo `data/raw/unlabeled.csv` içinden hasta seçer. Gerçek label yoktur.

Amaç:

```text
Deployment benzeri senaryoda ham etiketsiz hasta için tahmin ve açıklama üretebiliyor muyuz?
```

`--patient-position` ile farklı hasta seçilir.

Örnek:

```bash
python scripts/12_run_unlabeled_patient_llm_demo.py --patient-position 7 --no-save
```

`--no-save` kullanılırsa sonuç terminalde görülür ama report dosyaları değiştirilmez.

Bu, deneme yaparken temiz çalışmak için faydalıdır.

## 20. Script'ler Ne İşe Yarıyor?

| Script | Görev |
| --- | --- |
| `01_verify_preprocessing.py` | Preprocessing çıktısını notebook çıktısıyla doğrular |
| `02_verify_prediction.py` | Final model metriklerini yeniden üretir |
| `03_verify_explainability.py` | Tek hasta için SHAP explanation üretir |
| `04_verify_evidence.py` | SHAP sonucunu evidence packet'e dönüştürür |
| `05_verify_patient_pipeline.py` | Prediction + SHAP + evidence akışını birlikte test eder |
| `06_verify_prompt.py` | Evidence packet'ten prompt üretimini doğrular |
| `07_run_test_patient_demo.py` | Test hasta için prediction/evidence/prompt üretir |
| `08_run_test_patient_llm_demo.py` | Test hasta için LLM explanation + validation + revision çalıştırır |
| `09_save_preprocessor_artifact.py` | Fitted preprocessing artifact kaydeder |
| `10_run_saved_artifact_patient_demo.py` | Saved artifacts ile test hasta demosu çalıştırır |
| `11_run_unlabeled_patient_demo.py` | Unlabeled hasta için prediction/evidence/prompt üretir |
| `12_run_unlabeled_patient_llm_demo.py` | Unlabeled hasta için LLM explanation + validation + revision çalıştırır |
| `13_verify_validation.py` | Validator fixture testlerini çalıştırır |
| `14_audit_saved_explanations.py` | Kaydedilmiş açıklamaları toplu denetler |
| `15_run_gpt4o_subjective_evaluation.py` | Deterministic validation'dan geçen açıklamalara GPT-4o subjective evaluation uygular |

## 21. `src/` Dosyaları Ne İşe Yarıyor?

| Dosya | Görev |
| --- | --- |
| `src/preprocessing.py` | Ham ICU verisini model input formatına dönüştürür |
| `src/prediction.py` | Model ve threshold yükleme, prediction üretme |
| `src/explainability.py` | SHAP local explanation üretme |
| `src/evidence.py` | SHAP çıktısını structured evidence packet'e dönüştürme |
| `src/prompts.py` | Evidence packet'ten leakage-safe LLM prompt üretme |
| `src/llm.py` | GPT-4.1-mini explanation generation ve revision loop |
| `src/validation.py` | Deterministic validation checks ve scoring |
| `src/evaluator.py` | GPT-4o advisory subjective evaluation |
| `src/pipeline.py` | Preprocessing, prediction, SHAP ve evidence adımlarını birleştirme |

## 22. Notebook ve `.py` Ayrımı

Notebook'lar exploratory development için kullanıldı:

- EDA
- preprocessing kararları
- model karşılaştırmaları
- SHAP analizi
- evidence construction denemeleri
- prompt ve LLM reasoning denemeleri
- evaluation rubric geliştirme

Final pipeline ise `src/` ve `scripts/` altındadır.

Kısa ayrım:

```text
notebooks/ → kararların nasıl geliştirildiğini gösterir
src/       → final reusable kodu içerir
scripts/   → final pipeline'ı çalıştırır ve doğrular
reports/   → sonuçları, notları, auditleri ve değerlendirmeleri saklar
```

Bu ayrım raporda şöyle anlatılabilir:

```text
The notebooks document exploratory development, while the src modules and scripts implement the final reproducible pipeline.
```

## 23. Önemli Mimari Kararlar

### 23.1 Neden saved preprocessor artifact?

Başta preprocessing her seferinde training split üzerinde yeniden fit ediliyordu. Bu araştırma aşaması için uygundu ama deployment benzeri kullanım için uygun değildir.

Bu yüzden fitted preprocessing artifact kaydedildi:

```text
models/icu_preprocessor.pkl
```

Böylece yeni hasta geldiğinde:

```text
fit değil, sadece transform
```

uygulanır.

### 23.2 Neden true label prompt'tan çıkarıldı?

Çünkü açıklama model evidence'ına dayanmalıdır. Gerçek label prompt'ta olursa LLM açıklamayı outcome'a göre yazabilir. Bu label leakage olur.

Bu nedenle:

```text
y_true evidence packet'te kalabilir
y_true prompt'a gitmez
```

### 23.3 Neden deterministic validator?

Çünkü birçok hata deterministik olarak kontrol edilebilir:

- forbidden phrase
- leakage
- probability mismatch
- missing caution
- missing section
- exact feature hallucination
- exact direction flip

Bu hataları GPT-4o'ya sormak yerine doğrudan evidence packet'e göre kontrol etmek daha güvenilir ve tekrarlanabilir.

### 23.4 Neden GPT-4o tamamen kaldırılmadı?

Çünkü bazı kalite boyutları subjektiftir:

- clinical plausibility
- clarity

Bu boyutları regex ile adil şekilde ölçmek zordur. Bu yüzden GPT-4o advisory evaluator olarak kaldı.

### 23.5 Neden alias-aware caution matching?

Exact feature name aramak bazı doğru açıklamaları yanlış fail yapıyordu.

Örnek:

```text
Feature: icu_id
Explanation: The ICU unit identifier should be interpreted cautiously...
```

Bu açıklama doğruydu ama ilk exact-match validator `icu_id` literal string'i geçmediği için fail diyebiliyordu.

Çözüm:

```text
icu_id veya ICU unit identifier gibi approved alias kabul edildi.
```

Ama bu sadece caution-flagged küçük feature grubu için yapıldı. Feature grounding ve direction checks hâlâ exact-match v1 olarak bırakıldı.

## 24. Çıktılar Nasıl Okunur?

### Prediction JSON

Örnek dosya:

```text
reports/08_unlabeled_demo/unlabeled_patient_0_prediction.json
```

Şunu gösterir:

```text
death_probability
prediction
threshold
```

### Evidence JSON

Örnek:

```text
*_evidence.json
```

Şunları gösterir:

- risk artıran SHAP evidence
- risk azaltan SHAP evidence
- feature value
- clinical meaning
- caution flags

### Prompt TXT

Örnek:

```text
*_prompt.txt
```

LLM'e gönderilen temiz prompt'tur. True label veya case type içermemelidir.

### Explanation TXT

Örnek:

```text
*_llm_explanation.txt
```

LLM'in ilk ürettiği açıklamadır.

### Revised Explanation TXT

Örnek:

```text
*_llm_revised_explanation.txt
```

Validator fail sonrası revize edilmiş açıklamadır.

### Validation JSON

Örnek:

```text
*_llm_validation.json
```

Validator report içerir. Burada şunlara bakılır:

```text
passed
revision_required
deterministic_validation_score
checks
revision_feedback
```

### Audit CSV

Örnek:

```text
reports/09_validation_audit/validation_audit_summary.csv
```

Tüm saved explanations için toplu validation tablosudur.

### GPT-4o Evaluation CSV

Örnek:

```text
reports/10_gpt4o_evaluation/gpt4o_subjective_evaluation_summary.csv
```

Sadece deterministic validation'dan geçen açıklamalar için clinical plausibility ve clarity skorlarını içerir.

## 25. Bir Hasta İçin Sonuç Nasıl Yorumlanır?

Örnek prediction:

```json
{
  "death_probability": 0.9919,
  "prediction": 1.0,
  "threshold": 0.5
}
```

Yorum:

```text
Model bu hasta için mortalite olasılığını 0.9919 olarak hesapladı.
Bu değer 0.50 threshold üstünde olduğu için prediction = 1 oldu.
```

Sonra evidence packet'e bakılır:

```text
risk-increasing evidence → model riskini artıran SHAP katkıları
risk-decreasing evidence → model riskini azaltan SHAP katkıları
```

LLM explanation bu evidence'ı insan okunabilir dile çevirir.

Validator şunu kontrol eder:

```text
LLM bu evidence'ı doğru, dikkatli ve leakage olmadan anlattı mı?
```

## 26. Projeyi Savunurken Kullanılabilecek Kısa Anlatım

Bu projede önce ICU mortality prediction için LightGBM tabanlı güçlü bir model geliştirdim. Modelin tek başına olasılık üretmesini yeterli görmedim; bu yüzden her hasta için local SHAP explanation ürettim. SHAP çıktısını doğrudan LLM'e vermek yerine structured evidence packet formatına dönüştürdüm. Bu packet içinde risk artıran ve azaltan faktörler, SHAP yönleri, feature değerleri, klinik anlamlar ve caution flag'ler yer aldı.

Daha sonra bu evidence packet'ten leakage-safe bir prompt oluşturdum. True label ve TP/FN/FP/TN bilgilerini prompt'tan özellikle çıkardım, çünkü açıklamanın gerçek sonuçtan etkilenmesini istemedim. GPT-4.1-mini bu evidence'a dayalı açıklama üretti.

LLM açıklamasını doğrudan final kabul etmedim. `src/validation.py` içinde deterministik validator geliştirdim. Bu validator forbidden wording, true-label leakage, probability consistency, caution mentions, section structure, feature grounding ve SHAP direction consistency kontrolleri yapıyor. Hata varsa revision loop açıklamayı structured feedback ile düzeltiyor.

Son olarak tüm kaydedilmiş açıklamaları audit script'i ile toplu denetledim ve deterministic validation'dan geçen açıklamalar için GPT-4o'yu sadece clinical plausibility ve clarity gibi subjektif boyutlarda advisory evaluator olarak kullandım. Böylece hard safety checks deterministik, subjektif kalite değerlendirmesi ise ayrı ve advisory kaldı.

## 27. Bilinen Sınırlamalar

Bu proje güçlü bir end-to-end pipeline kursa da bazı sınırlamalar bilinçli olarak not edilmiştir.

Feature grounding ve direction consistency v1 exact feature-name matching kullanır. LLM teknik feature adı yerine doğal klinik ifade kullanırsa bu check'ler o kısmı tam değerlendiremeyebilir.

Caution matching alias-aware hale getirilmiştir, fakat sadece küçük caution-flagged feature grubu için uygulanır. Daha geniş alias-aware grounding gelecekte eklenebilir.

GPT-4o evaluator subjektif boyutlar için kullanılır, ama klinik uzman değerlendirmesinin yerini tutmaz.

Model prediction klinik karar değildir. Bu proje araştırma ve eğitim amaçlıdır.

## 28. En Önemli Öğrenilen Dersler

Bu projede teknik olarak en önemli öğrenimler şunlardır:

- XAI sadece grafik üretmek değildir; açıklamanın hangi evidence'a dayandığını yapılandırmak gerekir.
- LLM açıklamaları akıcı olabilir ama denetlenmeden güvenilir kabul edilmemelidir.
- True label leakage çok kolay oluşabilir; prompt ve evidence packet ayrımı bilinçli yapılmalıdır.
- Deterministic validator, LLM judge'a göre daha güvenilir hard gatekeeper olabilir.
- GPT-4o gibi güçlü modeller bile evaluator olarak false positive üretebilir.
- Caution flag'ler model açıklamasında klinik nedensellik ile model pattern'i arasındaki farkı korur.
- Saved artifacts, pipeline'ı notebook dışına taşıyıp deployment benzeri hale getirir.
- Audit CSV, açıklama kalitesini tek tek örnekler yerine sistematik olarak raporlamayı sağlar.

## 29. Son Özet

Bu proje baştan sona şu fikri uygular:

```text
Predict → Explain → Ground → Validate → Revise → Audit → Evaluate
```

Model tahmin üretir. SHAP bu tahmini açıklar. Evidence packet açıklamayı yapılandırır. Prompt LLM'e güvenli bilgi verir. LLM okunabilir açıklama üretir. Validator açıklamayı denetler. Revision loop hataları düzeltir. Audit tüm açıklamaları sistematik şekilde kontrol eder. GPT-4o ise sadece subjektif kalite değerlendirmesi sağlar.

Bu nedenle proje sadece bir mortality prediction modeli değil, açıklamaları evidence-grounded ve auditable hale getiren bir XAI pipeline'dır.
