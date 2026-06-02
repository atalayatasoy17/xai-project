# Çalışma Kağıdı — Açıklanabilir ICU Mortalite Tahmin Pipeline'ı

Bu doküman projeyi **baştan sona, öğrenme amaçlı** anlatır. Her aşama için üç soruya cevap verir:
**ne yaptık**, **neden yaptık** (hangi alternatifi neden reddettik), **nasıl çalışıyor** (kod mantığı + dosya).

Amaç sadece "ne var" demek değil; kararların arkasındaki düşünceyi de aktarmak.

---

## 0. Büyük Resim

### Problem ne?
Yoğun bakımdaki (ICU) bir hastanın **hastane içinde ölüp ölmeyeceğini** tahmin etmek (`hospital_death`, ikili sınıflandırma). Veri: WiDS Datathon 2020, ~91.713 hasta.

### Bu neden sadece "model kurma" projesi değil?
Klinik bir modelin tek başına "%92 risk" demesi yeterli değildir. Doktor **neden** öyle dediğini görmek ister. Bu yüzden proje iki katmanlı bir hedefe sahip:

1. **Tahmin et** — iyi bir mortalite modeli kur.
2. **Açıkla** — her tahmini, izlenebilir ve kanıta dayalı şekilde anlat.

### Projenin tek cümlelik felsefesi
> Model açıklamaları kanıta dayalı (evidence-grounded), denetlenebilir (auditable) ve desteksiz LLM çıkarımına karşı korunmuş (protected) olmalıdır.

Bu cümle her tasarım kararının arkasındaki pusuladır. Bir şey bu üç ilkeyi güçlendiriyorsa yaptık, zayıflatıyorsa reddettik.

### Akışın kuşbakışı hali
```text
ham hasta verisi
→ preprocessing (öğrenilmiş artifact)
→ LightGBM mortalite tahmini
→ yerel SHAP açıklaması
→ yapılandırılmış kanıt paketi (evidence packet)
→ LLM açıklama prompt'u
→ GPT-4.1-mini açıklaması
→ deterministik doğrulama (validator)
→ gerekirse revizyon
→ doğrulama denetimi (audit)
→ GPT-4o öznel değerlendirme (advisory)
```

### Mimari neden böyle iki ayrı dünyaya bölünmüş?
Projede iki tür "iş" var:
- **Nesnel / kurallaştırılabilir işler** (etiket sızıntısı var mı, olasılık doğru yazılmış mı, bölümler tam mı): bunları **deterministik kod** yapar — her zaman aynı sonucu verir, halüsinasyon yapmaz.
- **Öznel işler** (açıklama klinik olarak mantıklı mı, akıcı mı): bunları **LLM** değerlendirir — ama sadece tavsiye verir, karar veremez.

Bu ayrım projenin **ana katkısıdır** (Bölüm 9'da derinleşeceğiz).

---

## 1. Veri ve Temel Zorluk

### Ne yaptık
WiDS 2020 ICU verisini kullandık. Hedef `hospital_death`. ~180 ham sütun (vital bulgular, laboratuvar değerleri, APACHE skorları, demografik bilgiler).

### Neden önemli bir zorluk var: sınıf dengesizliği
Ölüm oranı sadece **~%8.6**. Yani veri çok dengesiz. Bu, model değerlendirmesini doğrudan etkiler:
- **Accuracy yanıltıcıdır.** "Herkes yaşayacak" diyen aptal bir model bile ~%91 accuracy alır ama hiçbir ölümü yakalayamaz.
- Bu yüzden **AUPRC, recall, F1, confusion matrix** birlikte bakılması gerekir.

Bu karar (accuracy'ye güvenmeme) ileride model seçimini şekillendirdi.

### Kavram: neden "leakage" (sızıntı) bu projede tekrar tekrar karşımıza çıkıyor?
Leakage = modelin/açıklamanın **görmemesi gereken bilgiyi görmesi**. İki yerde karşımıza çıktı:
1. **Veri leakage'i** — APACHE death probability sütunları (modelin kopya çekmesi gibi).
2. **Etiket leakage'i** — LLM açıklamasına gerçek sonucun sızması.

İkisini de ayrı ayrı engelledik (Bölüm 2 ve Bölüm 7).

---

## 2. Preprocessing — `src/preprocessing.py`

### Ne yaptık
Ham veriyi modelin anlayacağı 263 sütunluk sayısal bir matrise çevirdik. Adımlar:
- ID sütunlarını sil (`encounter_id`, `patient_id`, `hospital_id`) — tahmin gücü yok, ezberleme riski var.
- APACHE death probability sütunlarını sil — **bunlar zaten bir ölüm olasılığı; modele vermek kopya çekmek olur (leakage)**.
- Train'de %50'den fazla eksik olan sütunları at (74 sütun).
- Kalan eksikli sütunlar için **missingness indicator** üret (99 sütun) — "bu değer eksikti" bilgisi bazen başlı başına anlamlı.
- Sayısalları **train medyanı** ile doldur.
- Kategorikleri `Unknown` ile doldur, one-hot encode et.
- Train ve test sütun şemasını hizala.

### Neden bu kararlar — en kritik nokta: fit/transform ayrımı
**En önemli tasarım kararı:** Tüm bu kararlar (hangi sütun atılacak, medyanlar ne, kategoriler neler) **sadece train verisinden öğrenilir**.

Neden? Çünkü eğer medyanı tüm veriden (train+test) hesaplarsan, test bilgisi train'e sızar — bu da bir leakage'dir ve modelin gerçek performansını şişirir.

Çözüm: `ICUPreprocessor` sınıfı, scikit-learn mantığında çalışır:
- **`fit(train)`** → kararları öğrenir (hangi sütun, hangi medyan, hangi feature isimleri).
- **`transform(yeni_veri)`** → öğrenilmiş kararları uygular, **yeniden öğrenmez**.
- `fit_transform()` ikisini birleştirir.

`transform`'da kritik satır: `reindex(columns=self.feature_names_, fill_value=0)` — yeni veride eksik sütun varsa 0 ile doldurur, fazla sütun varsa atar. Böylece **her yeni hasta tam olarak modelin beklediği 263 sütun şemasına** oturur.

### Nasıl çalışıyor (deployment açısından)
Bu sınıf bir kez `fit` edilip **`models/icu_preprocessor.pkl`** olarak kaydedildi. Artık yeni/etiketsiz bir hasta geldiğinde preprocessing yeniden eğitilmez — kaydedilmiş artifact `transform` eder. Bu, projeyi "gerçek bir sistem" yapan şeydir.

### Doğrulama
`scripts/01_verify_preprocessing.py` — sınıfın ürettiği X_test'in, notebook'ta üretilen X_test ile **birebir aynı** olduğunu kanıtlar (shape 18343×263, sütunlar, değerler, y_test hepsi eşleşiyor).

---

## 3. Modelleme — `src/prediction.py` + `reports/01_modeling/`

### Ne yaptık
Birçok model denedik: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, dengelenmiş (class-weighted) versiyonlar, Optuna ile tune edilmiş versiyonlar.

Final model: **LightGBM Tuned Clean**, eşik (threshold) = 0.50.

Final test metrikleri:
| Metrik | Değer |
|---|---:|
| AUROC | 0.9019 |
| AUPRC | 0.5824 |
| Accuracy | 0.9013 |
| Precision | 0.4486 |
| Recall | 0.6286 |
| F1 | 0.5235 |

### Neden bu kararlar
- **Neden sadece accuracy'ye bakmadık:** Bölüm 1'deki dengesizlik. AUPRC, dengesiz problemde pozitif sınıfı (ölüm) yakalama kalitesini accuracy'den çok daha dürüst gösterir.
- **Neden eşik 0.50:** Klinik bağlamda **false negative (FN) tehlikelidir** — ölecek bir hastayı "yaşar" demek, yaşayacak birine "ölür" demekten daha riskli. 0.50 eşiği, bu bağlamda recall/FN dengesini makul tuttu.
- **Neden LightGBM:** Tablo verisinde güçlü, hızlı, ve SHAP TreeExplainer ile mükemmel uyumlu (Bölüm 4).

### Nasıl çalışıyor
`predict_mortality(model, X, threshold)`:
```text
olasılık = model.predict_proba(X)[:, 1]
tahmin   = (olasılık >= threshold) ? 1 : 0
```
Çıktı: `death_probability`, `prediction`, `threshold` içeren bir satır. Model + eşik diskten yüklenir (`load_model`, `load_threshold`).

### Doğrulama
`scripts/02_verify_prediction.py` — kaydedilmiş modelin notebook'taki metrikleri birebir ürettiğini gösterir.

---

## 4. SHAP Açıklanabilirlik — `src/explainability.py` + `reports/02_explainability/`

### Kavram: SHAP nedir?
SHAP, bir tahmine **her feature'ın ne kadar katkı yaptığını** sayısal olarak verir. Oyun teorisindeki Shapley değerlerine dayanır. Yorumu basit:
- **SHAP > 0** → o feature **riski artırdı**.
- **SHAP < 0** → o feature **riski azalttı**.
- **|SHAP| büyük** → o feature etkili.

### Ne yaptık
Final LightGBM için SHAP analizi:
- Global feature importance (hangi feature genelde önemli).
- Yerel (tek hasta) açıklamalar.
- TP / FN / FP / TN vaka analizleri (4 farklı tahmin tipi).
- Grup bazlı SHAP hata analizi.

### Neden TreeExplainer
LightGBM ağaç tabanlı bir model. SHAP'ın `TreeExplainer`'ı ağaç modelleri için **kesin ve hızlı** SHAP değeri hesaplar (örnekleme yapan diğer açıklayıcılardan daha güvenilir).

### Nasıl çalışıyor
`explain_patient(model, X_patient)`:
- Tek hasta satırı alır.
- TreeExplainer ile pozitif sınıf SHAP değerlerini hesaplar.
- Her feature için `[feature, value, shap_value, abs_shap_value, direction]` tablosu üretir.
- `|shap|` değerine göre sıralar (en etkili feature en üstte).

### Kritik kavram: caution (dikkat) gerektiren feature'lar
Bazı feature'lar yüksek SHAP'a sahip olsa da **klinik olarak doğrudan yorumlanmamalı**:
- `icu_id` → bu bir **hastane ünitesi kimliği**, klinik bir durum değil. Yüksek SHAP'ı "bu üniteye düşen hastalar genelde şöyle" gibi bir desenden gelebilir, hastanın kendi durumundan değil.
- Sıfır değerli vital bulgular (`d1_heartrate_min = 0`) → ya kayıt hatası ya da ekstrem klinik olay.
- Negatif `pre_icu_los_days` → zamanlama/veri kalitesi sorunu.

Bu feature'lar için sonradan **caution flag** ürettik (Bölüm 5). Bu, projenin "körlemesine güvenme" felsefesinin bir parçası.

---

## 5. Yapılandırılmış Kanıt Paketi — `src/evidence.py` + `reports/03_evidence/`

### Bu katman neden var? (En önemli tasarım fikirlerinden biri)
Ham SHAP tablosunu **doğrudan LLM'e vermedik**. Neden?
- Ham SHAP 263 satırlık, kalabalık, bağlamsız bir tablo.
- LLM'e ham veri verirsen üstüne **uydurma klinik yorum** ekleyebilir.

Çözüm: SHAP'ı önce **yapılandırılmış, klinik bağlamla zenginleştirilmiş bir JSON pakete** çeviriyoruz. Bu paket, model ile LLM arasında bir **köprü ve filtre** görevi görür.

### Ne yaptık — paket içeriği
Her paket şunları içerir:
- `prediction` → y_pred, y_proba, threshold (ve dahili analiz için y_true, prediction_type).
- `risk_increasing_evidence` → en çok riski artıran 8 feature.
- `risk_decreasing_evidence` → en çok riski azaltan 8 feature.
- Her kayıt için: `feature`, `value`, `shap_value`, `direction`, `clinical_meaning`, `caution_flags`.

### `clinical_meaning` nedir
`CLINICAL_MEANING_MAP` adlı bir sözlük, bazı feature'ları sade klinik dile çevirir:
- `age` → "older age is associated with higher mortality risk"
- `d1_spo2_min` → "low minimum oxygen saturation indicates hypoxemia"
- `icu_id` → "ICU unit identifier; may reflect unit-level patterns rather than patient-level clinical status"

Eğer bir feature sözlükte yoksa: `"No predefined clinical interpretation available."` yazılır. Bu çok önemli — LLM'e "bu feature için yorum uydurma" sinyali verir.

### `caution_flags` nasıl üretiliyor
`get_caution_flags(feature, value)` kural tabanlı çalışır:
- `icu_id` → her zaman "non-clinical identifier" flag'i.
- `pre_icu_los_days < 0` → negatif süre flag'i.
- Sıfır değerli vital → kayıt artefaktı flag'i.

### Nasıl çalışıyor (teknik detay)
`make_json_safe()` — numpy değerlerini (int64, float64) saf Python tiplerine çevirir, yoksa JSON'a yazılamaz. Küçük ama kritik bir köprü fonksiyonu.

### İç kayıt vs LLM'in gördüğü — kritik ayrım
> **evidence packet = bizim analiz kaydımız**
> **LLM prompt = modelin gördüğü bilgi**

`y_true` ve `prediction_type` paket **içinde kalır** (hata analizi için lazım), ama LLM prompt'una **gitmez** (Bölüm 7). Bu ayrımı aklında tut — leakage tartışmasının kalbi burası.

---

## 6. LLM Açıklama Üretimi — `src/prompts.py` + `src/llm.py` + `reports/06_llm_generation/`

### Ne yaptık
Evidence packet'ten doğal dilde, 5 bölümlü klinik açıklama ürettik:
1. Prediction summary
2. Main risk-increasing factors
3. Main risk-decreasing factors
4. Caution notes
5. Overall interpretation

Generator model: **gpt-4.1-mini** (ucuz, hızlı, bu iş için yeterli). Sıcaklık (temperature) = 0.0 → deterministik, tutarlı çıktı.

### Prompt nasıl kuruluyor — `build_explanation_prompt`
Prompt'a giren bilgiler:
- predicted label
- predicted probability
- threshold
- risk-increasing / risk-decreasing SHAP kanıtı (JSON)
- clinical meanings + caution flags

Üstüne **katı kurallar**: sadece kanıtı kullan, klinik fact uydurma, true label kullanma, birim ekleme, `clinical_meaning` yoksa yorum yapma, vs.

### Generator'ın sistem mesajı neden bu kadar katı?
LLM'ler **akıcı ama güvenilmez** olabilir — klinik olarak ikna edici görünüp aslında kanıtta olmayan detay ekleyebilir. Bu yüzden generator'ı bir "serbest yazar" değil, "kanıta bağlı bir raportör" olarak kısıtladık.

### Revizyon mekanizması — `revise_until_valid`
Üretilen açıklama doğrudan kabul edilmez. Şu döngü çalışır:
```text
açıklama üret
→ doğrula (validator)
→ sorun varsa: yapılandırılmış geri bildirimle revize ettir
→ tekrar doğrula
→ temiz olana kadar (veya max tur) devam
```
Bu, projenin **agentic review** (kendi kendini gözden geçiren) yönüdür.

### Tarihsel bir bulgu: notebook 08'deki keşif aşaması
İlk denemeler (notebook 08) iki LLM'li bir kurulumdu: **generator (gpt-4.1-mini) + evaluator (gpt-4o)**. Bu aşamada üç önemli şey öğrendik:
1. **LLM gerçek sonucu sızdırabiliyor** ("consistent with the true outcome" gibi).
2. **Agentic revizyon işe yarıyor** — TP vakası 3.8 → 4.0 → 4.7 puanla iyileşti.
3. **LLM evaluator de hata yapıyor** — bu en kritik bulgu (Bölüm 9).

Bu notebook bir **keşif aşamasıdır**; final pipeline daha sonra `src/` modüllerine taşındı.

---

## 7. Kritik Karar: Etiket Sızıntısının (Label Leakage) Önlenmesi

### Sorun neydi
İlk prompt'lar (eski notebook versiyonu) LLM'e şunları gönderiyordu:
```text
- Case type: TN        ← TP/FN/FP/TN, yani gerçek etiketi dolaylı kodluyor
- True label: 0        ← gerçek sonucun kendisi
```
Yani LLM'e açıklamayı yazmadan **önce cevabı veriyorduk**. Sistem mesajı "true label kullanma" dese de, bu bir çelişki: adama cevap anahtarını verip "bakma" demek gibi.

### Neden gerçek bir sorun
1. **Faithfulness bozulur** — açıklama modelin gerekçesini değil, gerçek sonucu yansıtmaya kayabilir.
2. **Gerçek hayatta o etiket yok** — yeni hasta geldiğinde sonucu bilmezsin (zaten o yüzden tahmin yapıyorsun).
3. **`case_type` bile sızdırıyor** — TP/FN = gerçek 1, FP/TN = gerçek 0.

### Nasıl çözdük
`src/prompts.py`'de `Case type` ve `True label` satırlarını prompt'tan **tamamen çıkardık**. Bunun yerine `Decision threshold` ekledik (bu modelin kendi çıktısı, sorun değil). 

Kritik incelik: `y_true` ve `prediction_type` evidence packet'te **kaldı** (Bölüm 5'teki ayrım) — çünkü onlar bizim hata analizimiz için lazım. Sadece **LLM'in gördüğü** kısımdan çıkarıldılar.

### Öğrenme noktası
Bu bir "utanılacak hata" değil, tam tersi: **biz yakaladık ve mimaride düzelttik**. Tezde "leakage riskini tespit edip düzelttim" demek, jürinin yakalamasından çok daha güçlü bir duruştur.

---

## 8. Deterministik Validator — `src/validation.py` (Projenin Kalbi)

### Bu katman neden var
Bölüm 6'da LLM açıklama üretiyor. Ama "akıcı görünüyor" ≠ "doğru". Bir açıklama:
- yasak/desteksiz kelime kullanabilir ("stable", "normal"),
- gerçek sonucu sızdırabilir,
- olasılığı yanlış yazabilir,
- caution flag'i atlayabilir,
- olmayan bir feature uydurabilir,
- SHAP yönünü ters çevirebilir.

Bunların hepsi **kuralla kontrol edilebilir** — LLM'e ihtiyaç yok. Bu yüzden deterministik bir validator kurduk.

### 7 kontrol — ne, neden
| Kontrol | Ne arar | Neden |
|---|---|---|
| `forbidden_phrases` | "stable", "normal", "favorable" gibi yorumlayıcı kelimeler | LLM kanıt dışına taşmasın |
| `true_label_leakage` | "died", "survived", "correct prediction" | gerçek sonuç sızmasın |
| `section_structure` | 5 bölüm var mı | açıklama standart kalsın |
| `prediction_consistency` | yazılan olasılık ≈ paketteki olasılık (±0.01) | LLM sayıyı yanlış aktarmasın |
| `caution_mentions` | flag'li feature dikkatli yorumlanmış mı | problemli değişken düz gerçek gibi sunulmasın |
| `feature_grounding` | açıklamadaki feature paketten mi | uydurma feature yakalansın |
| `direction_consistency` | "riski artırdı" denilen feature'ın SHAP'i pozitif mi | yön hatası yakalansın |

### Skorlama nasıl çalışıyor
Validator pass/fail'in ötesinde **1-5 arası deterministik skor** üretir. Ama sadece **nesnel ölçülebilen** 3 rubric boyutunu kapsar:
- faithfulness_no_hallucination (ağırlık 0.30)
- caution_awareness (0.20)
- completeness (0.15)

Skor formülü (sadece kapsanan boyutlar üzerinden normalize):
```text
skor = (0.30/0.65)*faithfulness + (0.20/0.65)*caution + (0.15/0.65)*completeness
```
`clinical_plausibility` ve `clarity` burada **ölçülmez** — onlar öznel, LLM judge'a bırakılır (Bölüm 9).

Her kontrolün ihlal sayısına göre puan düşer: `skor = 5 - ihlal_sayısı` (en az 1).

### Revizyon köprüsü
Validator sadece "hata var" demez, **ne yapılacağını** söyler. `revision_feedback` listesi üretir ("Remove unsupported wording: favorable", "Mention caution for: icu_id"). Bu liste revize prompt'una girer. Yani validator ↔ generator arası **kapalı bir geri besleme döngüsü** var.

### Kritik karar: caution check'te alias-aware eşleştirme
**Sorun:** Caution check başta feature'ın **kod ismini** (`icu_id`) literal arıyordu. Ama LLM klinik dil kullanıyor ("ICU unit identifier"). Sonuç: doğru verilmiş caution'lar "eksik" sanılıyor → **false positive** → gereksiz revizyon → açıklamaya kod ismi sokuluyor.

**Çözüm (alias-aware):** Caution-flag'li az sayıdaki feature için bir alias sözlüğü (`CAUTION_FEATURE_ALIASES`) ekledik:
- `icu_id` → "icu_id", "ICU unit identifier", "unit identifier", "unit-level"
- `d1_heartrate_min` → "minimum heart rate", "heart rate" ...

Artık kod ismi **VEYA** klinik alias kabul ediliyor. İki ek incelik:
1. Arama sadece **"Caution notes" bölümünde** yapılıyor (precision için).
2. "feature kimliği" ile "caution dili" **ayrı koşullar** — karıştırılmıyor (yoksa "data quality" deyince feature var sanılırdı = false pass).

**Neden sadece caution'a uyguladık, grounding/direction'a değil:** Caution'da exact-match hatası **aktif zarar** veriyor (yanlış revizyon tetikliyor). Grounding/direction'da exact-match hatası **pasif** (sessizce kaçırır, kimseye zarar vermez). Önce aktif zararı düzelttik.

### Fixture testleri — "validator'ı doğrulamak"
`scripts/13_verify_validation.py` — validator'ın kendisini test eder. Kontrollü örnekler:
- iyi açıklama → geçer (5.0)
- uydurma feature → yakalanır
- yön hatası → yakalanır
- true label sızıntısı → yakalanır
- eksik bölüm → yakalanır
- yanlış olasılık → yakalanır
- alias ile doğru caution → geçer (false positive düzeltmesinin kanıtı)

Bu, "validator gerçekten çalışıyor mu" sorusuna kanıt sunar.

### Audit — toplu denetim
`scripts/14_audit_saved_explanations.py` — kaydedilmiş tüm açıklamaları validator'dan geçirip tek CSV üretir. Güncel sonuç: 10 açıklama denetlendi, 7 geçti, 3 desteksiz kelime yüzünden kaldı, tüm revize edilmişler geçti.

### Önemli dürüst bulgu: exact-match'in sınırı
Gerçek açıklamalarda `ungrounded_features` ve `direction_errors` hep boş çıktı. Bu **"hiç hata yok" demek değil**, "exact-match kontrol edemedi" demek — çünkü LLM kod ismi yerine klinik dil kullanıyor. Bu sınır notlarda dürüstçe yazıldı; alias-aware v2'nin motivasyonu bu.

---

## 9. Hibrit Değerlendirme: GPT-4o'nun Dar Rolü — `src/evaluator.py` + `reports/10_gpt4o_evaluation/`

### Ana karar: GPT-4o neden her şeyi yapmıyor
Notebook 08'de gpt-4o **tüm rubric'i** puanlıyordu ve **revizyon tetikleyebiliyordu**. Ama bir sorun keşfettik (aşağıda). Bu yüzden final pipeline'da gpt-4o'nun rolünü **daralttık**.

### Smoking-gun bulgusu: evaluator halüsinasyonu
TP strict-revised vakasında gpt-4o şöyle dedi:
> "it uses the true label to justify the prediction" (açıklama gerçek etiketi kullanmış)

**Ama:** açıklama metninde "died/survived/true label" **hiç geçmiyordu**. Deterministik kontrol `forbidden_phrases: []` (0 ihlal) buldu. Yani gpt-4o **olmayan bir ihlali uydurdu** — bir halüsinasyon.

Bu, projenin en güçlü bilimsel argümanıdır: **LLM judge'lar güvenilmez olabilir, bu yüzden nesnel kontroller deterministik olmalı.**

### Çözüm: iki katmanlı (hibrit) değerlendirme
| Katman | Kim | Ne yapar | Yetki |
|---|---|---|---|
| **Deterministik validator** | kod | leakage, wording, grounding, direction, caution, prediction, section | **HARD gate** — pass/fail kararını verir |
| **GPT-4o evaluator** | gpt-4o | sadece clinical_plausibility + clarity | **advisory** — sadece skor verir, karar veremez |

GPT-4o neden hâlâ var? Çünkü "klinik mantıklı mı" ve "akıcı mı" gerçekten **öznel** — bunları regex ile ölçemezsin. Ama gpt-4o'ya sadece bu **iki dar boyutu** bırakarak halüsinasyon yüzeyini minimuma indirdik (5 boyut yerine 2).

### Hibrit skor
Final skor, deterministik 3 boyut + gpt-4o'nun 2 boyutunu orijinal rubric ağırlıklarıyla birleştirir:
```text
faithfulness(0.30) + plausibility(0.25) + caution(0.20) + completeness(0.15) + clarity(0.10)
```
Güncel sonuç: 7 doğrulanmış açıklama değerlendirildi, hibrit skorlar 4.65–4.90 arasında.

### Faz 2.5'i neden yapmadık (bilinçli karar)
"Validator skorlarını gpt-4o skorlarıyla karşılaştıralım" diye bir fikir vardı. Yapmadık çünkü: **ikisi farklı boyutları ölçüyor** (nesnel vs öznel), "uyum oranı" elmayla armut kıyaslaması olurdu. Ayrıca karar zaten verilmişti (gpt-4o dar kapsamda). Bunun yerine halüsinasyon bulgusunu tek vaka örneğiyle anlatmak daha güçlü.

---

## 10. Uçtan Uca Pipeline ve Demolar — `src/pipeline.py` + scripts

### `run_patient_pipeline` — her şeyi birleştiren fonksiyon
Tek hasta için tüm zinciri çalıştırır:
```text
preprocessor.transform → predict_mortality → explain_patient → build_evidence_packet
```
Çıktı: işlenmiş hasta, tahmin, yerel açıklama, evidence packet.

### Demolar — iki senaryo
1. **Etiketli test hastası** (`scripts/08`) — modelin doğruluğunu kontrol edebildiğimiz, etiketi bilinen hasta.
2. **Etiketsiz hasta** (`scripts/12`) — **gerçek deployment senaryosu**: sonucu bilinmeyen yeni hasta. `--patient-position` ile hasta seçilir, `--no-save` ile kaydetmeden test edilir.

### Neden iki demo
Etiketli demo "pipeline doğru mu" sorusunu cevaplar (notebook'la karşılaştırılabilir). Etiketsiz demo "gerçekte işe yarıyor mu" sorusunu cevaplar (etiket yokken de açıklama + doğrulama üretebiliyor).

### Kaydedilmiş artifact'ler
`scripts/09` preprocessor'ı `.pkl` olarak kaydeder. `scripts/10` ve `11` bu kaydedilmiş artifact'lerle çalışır — preprocessing'i yeniden eğitmeden. Bu "deployment-style" yaklaşımdır.

---

## 11. Aldığımız Kritik Kararlar — Özet Tablo

| Karar | Ne seçtik | Neden | Reddettiğimiz alternatif |
|---|---|---|---|
| Preprocessing | fit/transform ayrımı | train→test leakage'i önlemek | tüm veriden medyan (kolay ama sızıntılı) |
| Model metriği | AUPRC + recall + F1 | sınıf dengesizliği | sadece accuracy (yanıltıcı) |
| Eşik | 0.50 | FN'i azaltmak (klinik risk) | daha yüksek eşik (FN artar) |
| SHAP → LLM | önce evidence packet | LLM'i kanıta bağlamak | ham SHAP'ı doğrudan vermek (uydurma riski) |
| Prompt etiketi | true_label/case_type çıkarıldı | faithfulness + gerçekçilik | "kullanma" deyip yine de göndermek |
| Validation | deterministik hard gate | tekrarlanabilir, halüsinasyonsuz | sadece LLM judge (güvenilmez) |
| Caution eşleştirme | alias-aware | false positive'i önlemek | exact-match (kod ismi sokuyor) |
| GPT-4o rolü | sadece plausibility+clarity, advisory | halüsinasyon yüzeyini küçültmek | gpt-4o her şeyi puanlasın (notebook 08'de hata yaptı) |

---

## 12. Kavram Sözlüğü

- **SHAP**: Her feature'ın bir tahmine katkısını ölçen yöntem. Pozitif = riski artırdı, negatif = azalttı.
- **Faithfulness (sadakat)**: Açıklamanın, modelin gerçek gerekçesini ne kadar doğru yansıttığı.
- **Leakage (sızıntı)**: Model/açıklamanın görmemesi gereken bilgiyi görmesi. Veri leakage'i (APACHE prob) ve etiket leakage'i (true label) olarak ikiye ayrılır.
- **fit/transform**: `fit` train'den öğrenir, `transform` öğrenileni uygular. Ayrımı korumak leakage'i engeller.
- **Evidence packet**: Ham SHAP'ın klinik bağlamla zenginleştirilmiş, yapılandırılmış JSON hali. Model-LLM köprüsü.
- **Caution flag**: Bir feature'ın dikkatle yorumlanması gerektiğini belirten işaret (örn. icu_id non-klinik).
- **LLM-as-judge**: Bir LLM'in başka bir LLM'in çıktısını değerlendirmesi. Güçlü ama halüsinasyona açık.
- **Hallucination (halüsinasyon)**: LLM'in olmayan bir şeyi varmış gibi üretmesi. Burada: olmayan bir ihlali "var" demesi.
- **Deterministik**: Aynı girdiye her zaman aynı çıktıyı veren. Regex/kural tabanlı kontroller deterministiktir; LLM değildir.
- **Hard gate vs advisory**: Hard gate kararı verir (pass/fail). Advisory sadece görüş bildirir, karar veremez.
- **Agentic review**: Sistemin kendi çıktısını üretip, doğrulayıp, gerekirse düzeltmesi — bir "ajan" gibi davranması.

---

## 13. "Jüri Sorarsa" — Hazır Cevaplar

**S: Neden accuracy yüksek ama precision düşük?**
C: Sınıf dengesizliği (%8.6 ölüm). Accuracy çoğunluk sınıfından şişer; biz AUPRC/recall'a baktık. Düşük precision, klinikte FN'i azaltmak için kabul edilen bir takas.

**S: LLM açıklamasının gerçek sonuçtan etkilenmediğini nereden biliyorsunuz?**
C: İki katmanlı koruma: (1) prompt'tan true_label/case_type çıkarıldı, (2) deterministik validator true-label leakage'i tarıyor. Üstelik bu riski biz tespit edip düzelttik.

**S: Neden hem deterministik validator hem GPT-4o var?**
C: İş bölümü. Nesnel kontroller (leakage, sayı, yapı) deterministik — tekrarlanabilir ve halüsinasyonsuz. Öznel kontroller (klinik mantık, akıcılık) gpt-4o — ama sadece tavsiye, karar deterministik validator'da.

**S: GPT-4o güvenilmezse neden hâlâ kullanıyorsunuz?**
C: Sadece regex ile ölçülemeyen 2 öznel boyutta, üstelik advisory olarak. Hard kararı asla vermiyor. Halüsinasyon riskini bu daraltma ile minimuma indirdik.

**S: feature_grounding hep boş çıkıyor, işe yaramıyor mu?**
C: Exact-match v1 olduğu için, LLM klinik dil kullanınca eşleşme bulamıyor. Bu "hata yok" değil "ölçemedi" demek — notlarda dürüstçe belirtildi. Alias-aware v2 gelecek iş.

**S: icu_id neden özel olarak işaretli?**
C: Bu klinik bir değer değil, ünite kimliği. Yüksek SHAP'ı hasta durumundan değil, ünite-seviyesi desenden gelebilir. Bu yüzden caution flag'li ve dikkatle yorumlanıyor.

---

## 14. Dosya Haritası — Hangi Kod Nerede

| Katman | Modül | Doğrulama/Demo Script |
|---|---|---|
| Preprocessing | `src/preprocessing.py` | `scripts/01`, `scripts/09` |
| Tahmin | `src/prediction.py` | `scripts/02` |
| SHAP | `src/explainability.py` | `scripts/03` |
| Evidence packet | `src/evidence.py` | `scripts/04` |
| Prompt | `src/prompts.py` | `scripts/06` |
| LLM üretim + revizyon | `src/llm.py` | `scripts/08`, `scripts/12` |
| Deterministik validator | `src/validation.py` | `scripts/13`, `scripts/14` |
| GPT-4o değerlendirme | `src/evaluator.py` | `scripts/15` |
| Uçtan uca pipeline | `src/pipeline.py` | `scripts/05`, `scripts/10`, `scripts/11` |

Her aşamanın bir de `reports/<aşama>/` altında notları ve kaydedilmiş çıktıları vardır.

---

*Bu çalışma kağıdı projenin kendi kod ve raporlarına dayanır. Sayısal sonuçlar yazıldığı andaki audit/değerlendirme çıktılarını yansıtır; kod değişirse ilgili `reports/*/notes.md` ve CSV'lerle birlikte güncellenmelidir.*
