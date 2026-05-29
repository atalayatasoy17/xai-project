# Modeling Notes

## Amaç

Bu aşamada amaç, yoğun bakım hastalarında hastane içi ölüm riskini tahmin eden bir sınıflandırma modeli kurmak, farklı model ailelerini karşılaştırmak ve final model seçimini metriklere dayalı olarak gerekçelendirmektir.

Problem klinik risk tahmini olduğu için sadece accuracy'ye bakılmadı. Veri dengesiz olduğu için özellikle şu metrikler dikkate alındı:

- **AUPRC:** Pozitif sınıfın az olduğu durumlarda modelin ölüm sınıfını ne kadar iyi ayırdığını daha iyi gösterir.
- **AUROC:** Genel ayrıştırma gücünü gösterir; destekleyici metrik olarak kullanıldı.
- **Recall:** Gerçek ölüm vakalarının ne kadarını yakaladığımızı gösterir. Klinik bağlamda false negative kaçırmak önemli olduğu için kritik bir metriktir.
- **Precision:** Ölüm riski dediğimiz hastaların ne kadarının gerçekten pozitif olduğunu gösterir.
- **F1:** Precision ve recall dengesini özetler.
- **Confusion matrix:** TP, FP, FN, TN sayılarını doğrudan görerek kararın klinik etkisini yorumlamak için kullanıldı.

## İzlenen Modelleme Sırası

Önce basit ve yorumlanabilir bir baseline kuruldu, sonra daha güçlü ağaç tabanlı modeller denenerek performans karşılaştırıldı.

Denediğimiz ana modeller:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- XGBoost + imbalance weighting
- LightGBM + imbalance weighting
- Optuna ile tuned XGBoost
- Optuna ile tuned LightGBM

Bu sıra bilinçli seçildi: önce basit baseline ile referans seviye görüldü, sonra non-linear modellerin katkısı değerlendirildi.

## Imbalance Kararı

Pozitif sınıf az olduğu için imbalance ayrıca incelendi. XGBoost tarafında `scale_pos_weight`, LightGBM tarafında class-weight/imbalance yaklaşımı denendi.

Balanced modeller recall'u artırdı:

- XGBoost Balanced recall: **0.7100**
- LightGBM Balanced recall: **0.7757**

Fakat bu artış yüksek false positive maliyetiyle geldi:

- XGBoost Balanced FP: **2022**
- LightGBM Balanced FP: **2541**

Bu nedenle balanced modeller ölüm vakalarını daha fazla yakalasa da precision ve genel denge tarafında final seçim için en uygun modeller olmadı. Bu sonuç, imbalance yaklaşımının faydalı olduğunu ama tek başına final model seçimi için yeterli olmadığını gösterdi.

## Hyperparameter Tuning

Final aday modeller için Optuna ile hiperparametre optimizasyonu yapıldı. Optimizasyon metriği olarak **AUPRC** seçildi, çünkü pozitif sınıf az ve projenin hedefi ölüm sınıfını doğru yakalamak.

Tuning sırasında validation setini doğrudan modele ezberletmemek için daha temiz bir yaklaşım izlendi:

- `X_train` içinden `X_tr / X_val` ayrıldı.
- Optuna tuning işlemi `X_tr` üzerinde 3-fold cross-validation ile yapıldı.
- `X_val`, threshold analizi için ayrı tutuldu.
- Final test değerlendirmesi yalnızca `X_test` üzerinde yapıldı.

Bu akış leakage riskini azaltır ve model seçimi ile final test değerlendirmesini daha savunulabilir hale getirir.

## Threshold Kararı

Tuned XGBoost ve tuned LightGBM için threshold sweep yapıldı. Threshold değişince AUROC ve AUPRC değişmez; çünkü bu metrikler olasılık sıralamasına bakar. Threshold değişimi daha çok precision, recall, F1 ve confusion matrix üzerinde etkilidir.

LightGBM için F1'e göre 0.60 threshold iyi görünse de:

- Threshold 0.60:
  - Precision: **0.5144**
  - Recall: **0.5534**
  - F1: **0.5332**
  - FN: **707**
  - TP: **876**

- Threshold 0.50:
  - Precision: **0.4486**
  - Recall: **0.6286**
  - F1: **0.5235**
  - FN: **588**
  - TP: **995**

F1 açısından 0.60 biraz daha yüksek olsa da klinik bağlamda false negative sayısını azaltmak daha önemli kabul edildi. Bu nedenle final threshold olarak **0.50** seçildi.

Bu kararın anlamı: model, biraz daha fazla false positive üretmeyi kabul ederek daha fazla gerçek ölüm vakasını yakalar.

## Final Model Seçimi

Final model olarak **LightGBM Tuned Clean, threshold = 0.50** seçildi.

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

XGBoost modelinin AUROC/AUPRC değerleri çok yakın ve bazı metriklerde az farkla daha yüksek görünmektedir:

- XGBoost AUPRC: **0.5835**
- Selected LightGBM AUPRC: **0.5824**

Ancak XGBoost threshold 0.50'de recall açısından zayıf kaldı:

- XGBoost recall: **0.3361**
- Selected LightGBM recall: **0.6286**

Bu fark klinik problem için önemlidir. XGBoost daha az false positive üretse de çok daha fazla gerçek ölüm vakasını kaçırmaktadır:

- XGBoost FN: **1051**
- Selected LightGBM FN: **588**

Bu nedenle final seçim sadece en yüksek AUPRC değerine göre değil, AUPRC + recall + FN dengesi birlikte düşünülerek yapıldı.

## Kaydedilen Çıktılar

Modelleme aşamasında oluşturulan ana çıktılar:

- `models/lgbm_tuned_clean.pkl`: final seçilen LightGBM modeli
- `models/lgbm_tuned_clean_threshold.json`: final threshold bilgisi
- `reports/01_modeling/final_model_comparison.csv`: tüm ana model sonuçlarının karşılaştırma tablosu

## Sonuç

Modelleme aşamasında final karar şu şekilde özetlenebilir:

> Final model olarak tuned LightGBM seçildi. Threshold 0.50 bırakıldı, çünkü bu eşik ölüm vakalarını yakalama açısından daha iyi recall ve daha düşük false negative sayısı sağladı. Projenin klinik risk tahmini bağlamında false negative vakaları kaçırmak kritik olduğundan, karar yalnızca accuracy veya en yüksek AUPRC değerine göre değil, confusion matrix ve recall dengesiyle birlikte verildi.

Bu model sonraki aşamalarda SHAP explainability, evidence construction ve LLM tabanlı açıklama üretimi için temel model olarak kullanıldı.
