# **Vegetable Classification**

## 데이터셋

This project uses the [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset) from Kaggle.

* **Classes:** 15 vegetable types (`bean`, `bitter gourd`, `bottle gourd`, `brinjal`, `broccoli`, `cabbage`, `capsicum`, `carrot`, `cauliflower`, `cucumber`, `papaya`, `potato`, `pumpkin`, `radish`, `tomato`)
* **Total Images:** 21,000 images (`224×224`, `.jpg`)
* **Split:** Train (15,000) / Validation (3,000) / Test (3,000)

### 출처

If you use this dataset, please cite the original paper as requested by the authors:

* M. I. Ahmed, S. M. Mamun, and A. U. Z. Asif, "[DCNN-based vegetable image classification using transfer learning: A comparative study](https://www.researchgate.net/publication/352846889_DCNN-Based_Vegetable_Image_Classification_Using_Transfer_Learning_A_Comparative_Study)," in *2021 5th International Conference on Computer, Communication and Signal Processing (ICCCSP)*, IEEE, 2021, pp. 235–243.


## 입력 파라미터: 랜덤 파라미터 10개 생성
- `PARAMETER_SEED`: 파라미터용 시드
  + 별도로 랜덤 생성 후 각 파라미터 생성에서 공유
## 성능 비교: 각 모델별 정확도가 가장 높은 5개의 파라미터 조합의 평균값 비교
### Logistic Regression
평균: 
 - Top 1: params={'C': np.float64(0.3502718917554013), 'max_iter': 1553} | test_accuracy=0.6713
 - Top 2: params={'C': np.float64(0.10477323365227816), 'max_iter': 667} | test_accuracy=0.6727
 - Top 3: params={'C': np.float64(0.25202399872719894), 'max_iter': 515} | test_accuracy=0.6707
 - Top 4: params={'C': np.float64(0.6041468348066917), 'max_iter': 1639} | test_accuracy=0.6653
 - Top 5: params={'C': np.float64(0.46737903782076806), 'max_iter': 1358} | test_accuracy=0.6697
### k-Nearest Neighbors (k-NN)
평균: 
-  Top 1: params={'n_neighbors': 11, 'p': 1, 'weights': 'distance'} | test_accuracy=0.8283
-  Top 2: params={'n_neighbors': 8, 'p': 1, 'weights': 'uniform'} | test_accuracy=0.8153
-  Top 3: params={'n_neighbors': 9, 'p': 2, 'weights': 'distance'} | test_accuracy=0.8040
-  Top 4: params={'n_neighbors': 17, 'p': 1, 'weights': 'distance'} | test_accuracy=0.7890
-  Top 5: params={'n_neighbors': 18, 'p': 1, 'weights': 'distance'} | test_accuracy=0.7830
### Support Vector Machine (SVM)
평균: 
-  Top 1: params={'C': np.float64(44.00139059238714), 'gamma': np.float64(0.0021602642622069383), 'kernel': 'rbf'} | test_accuracy=0.9573
-  Top 2: params={'C': np.float64(961.4375218857606), 'gamma': np.float64(0.0012564327892794409), 'kernel': 'rbf'} | test_accuracy=0.9560
-  Top 3: params={'C': np.float64(1.6777388393177377), 'gamma': np.float64(0.0008544397929536556), 'kernel': 'rbf'} | test_accuracy=0.9227
-  Top 4: params={'C': np.float64(34.881044844591194), 'gamma': np.float64(0.006034305320476808), 'kernel': 'rbf'} | test_accuracy=0.9270
-  Top 5: params={'C': np.float64(94.25660090175117), 'gamma': np.float64(0.0001330435200932225), 'kernel': 'rbf'} | test_accuracy=0.9183
### Decision Tree
평균: 
-  Top 1: params={'criterion': 'entropy', 'max_depth': 45, 'max_features': np.float64(0.48438184183808064), 'min_samples_leaf': 1} | test_accuracy=0.5847
-  Top 2: params={'criterion': 'entropy', 'max_depth': 40, 'max_features': np.float64(0.48590207748087033), 'min_samples_leaf': 1} | test_accuracy=0.5957
-  Top 3: params={'criterion': 'entropy', 'max_depth': 12, 'max_features': np.float64(0.647599603185538), 'min_samples_leaf': 1} | test_accuracy=0.5717
-  Top 4: params={'criterion': 'entropy', 'max_depth': 21, 'max_features': np.float64(0.8008884735754034), 'min_samples_leaf': 4} | test_accuracy=0.5693
-  Top 5: params={'criterion': 'gini', 'max_depth': 21, 'max_features': np.float64(0.32796342556421637), 'min_samples_leaf': 3} | test_accuracy=0.5753
### Random Forest
평균: 
-  Top 1: params={'max_depth': 47, 'max_features': np.float64(0.412868964648599), 'min_samples_leaf': 1, 'n_estimators': 182} | test_accuracy=0.8950
-  Top 2: params={'max_depth': 27, 'max_features': np.float64(0.6577668147382395), 'min_samples_leaf': 2, 'n_estimators': 196} | test_accuracy=0.8783
-  Top 3: params={'max_depth': 47, 'max_features': np.float64(0.49669890222806623), 'min_samples_leaf': 3, 'n_estimators': 165} | test_accuracy=0.8833
-  Top 4: params={'max_depth': 47, 'max_features': np.float64(0.48438184183808064), 'min_samples_leaf': 3, 'n_estimators': 93} | test_accuracy=0.8740
-  Top 5: params={'max_depth': 35, 'max_features': np.float64(0.6380442565692083), 'min_samples_leaf': 3, 'n_estimators': 65} | test_accuracy=0.8603
### Bagging Classifier
평균: 
-  Top 1: params={'base_max_depth': 30, 'max_features': np.float64(0.824542293027221), 'max_samples': np.float64(0.8397952957497588), 'n_estimators': 77} | test_accuracy=0.8657
-  Top 2: params={'base_max_depth': 21, 'max_features': np.float64(0.8577774811252881), 'max_samples': np.float64(0.66607900647939), 'n_estimators': 93} | test_accuracy=0.8647
-  Top 3: params={'base_max_depth': 26, 'max_features': np.float64(0.7623924077137898), 'max_samples': np.float64(0.897431177410142), 'n_estimators': 49} | test_accuracy=0.8653
-  Top 4: params={'base_max_depth': 26, 'max_features': np.float64(0.6969886825702467), 'max_samples': np.float64(0.5211763359571939), 'n_estimators': 96} | test_accuracy=0.8660
-  Top 5: params={'base_max_depth': 12, 'max_features': np.float64(0.7482854308468129), 'max_samples': np.float64(0.9903740279316868), 'n_estimators': 70} | test_accuracy=0.8277
### Extra Trees
평균: 
-  Top 1: params={'max_depth': 47, 'max_features': np.float64(0.412868964648599), 'min_samples_leaf': 1, 'n_estimators': 182} | test_accuracy=0.9093
-  Top 2: params={'max_depth': 27, 'max_features': np.float64(0.6577668147382395), 'min_samples_leaf': 2, 'n_estimators': 196} | test_accuracy=0.9027
-  Top 3: params={'max_depth': 47, 'max_features': np.float64(0.49669890222806623), 'min_samples_leaf': 3, 'n_estimators': 165} | test_accuracy=0.8987
-  Top 4: params={'max_depth': 42, 'max_features': np.float64(0.6410158363118194), 'min_samples_leaf': 3, 'n_estimators': 110} | test_accuracy=0.9013
-  Top 5: params={'max_depth': 47, 'max_features': np.float64(0.48438184183808064), 'min_samples_leaf': 3, 'n_estimators': 93} | test_accuracy=0.8960
### AdaBoost
평균: 
-  Top 1: params={'base_max_depth': 3, 'learning_rate': np.float64(0.040373588785906925), 'n_estimators': 104} | test_accuracy=0.4597
-  Top 2: params={'base_max_depth': 3, 'learning_rate': np.float64(0.1291856423208951), 'n_estimators': 48} | test_accuracy=0.4527
-  Top 3: params={'base_max_depth': 3, 'learning_rate': np.float64(1.901799945165288), 'n_estimators': 99} | test_accuracy=0.4057
-  Top 4: params={'base_max_depth': 2, 'learning_rate': np.float64(0.4078482868970378), 'n_estimators': 44} | test_accuracy=0.3650
-  Top 5: params={'base_max_depth': 3, 'learning_rate': np.float64(0.012515689160377982), 'n_estimators': 148} | test_accuracy=0.3650
### Gradient Boosting
평균: 
### Perceptron
평균: 
### Multi-Layer Perceptron (MLP)
평균: 
### Deep Neural Network (DNN)
평균: 
---
## **각 모델의 파라미터 및 특이점 설명**
### 1. 선형 · 거리 · 마진 계열

#### Logistic Regression
* **`C` (규제 강도의 역수):**
  * **증가 시:** 규제가 약해져 학습 데이터에 더 밀착(과적합 위험 증가).
  * **감소 시:** 규제가 강해져 가중치 크기를 억제(과소적합 위험 증가, 일반화 방향).
* **`penalty` (`l2`, `l1`, `elasticnet`, `none`):**
  * **`l2`:** 가중치를 고르게 작게 만들어 잡음·공선성에 비교적 안정적.
  * **`l1`:** 일부 가중치를 `0`으로 만들어 특성 선택(Feature Selection) 효과.
* **`max_iter`:** 최적화 반복 횟수. 수렴이 부족할 때 늘려야 함.
* C값이 크고, max_iter값이 작으면 학습이 완료되기 전 max_iter 값에 도달해서 학습을 중지함

#### k-Nearest Neighbors (k-NN)
* **`n_neighbors` (`k`, 이웃 수):**
  * **감소 시(예: `1`, `3`):** 소수 이웃만 참고해 결정 경계가 세밀해지고 **과적합·노이즈 민감도** 상승.
  * **증가 시:** 넓은 범위로 판단해 경계가 매끄러워지나, 과도하면 **과소적합** 발생.
* **`weights` (`uniform`, `distance`):**
  * **`uniform`:** 이웃 `k`개에 동일 가중치 부여.
  * **`distance`:** 가까운 이웃에 더 큰 가중치를 주어 국소적 영향력 확대.
* **`metric` (`euclidean`, `manhattan`, `minkowski` 등):** 거리 계산 기준. 특성 분포와 차원에 따라 이웃 판정 변화.

#### Support Vector Machine (SVM)
* **`C` (규제 파라미터):**
  * **증가 시:** 오분류를 줄이려 경계를 복잡하게 잡음 → **과적합 위험 상승**.
  * **감소 시:** 마진을 넓게 확보 → **과소적합 위험 상승, 일반화 경향**.
* **`kernel` (`linear`, `rbf`, `poly`, `sigmoid`):**
  * **`linear`:** 직선/초평면 경계만 형성.
  * **`rbf` / `poly`:** 비선형 경계 형성 가능하나, 랜덤 데이터에서는 불필요한 복잡도(과적합) 주의.
* **`gamma` (`rbf` / `poly` 커널 영향 반경):**
  * **증가 시:** 한 샘플의 영향 범위가 좁아져 경계가 굴곡짐 → **과적합 경향**.
  * **감소 시:** 영향 범위가 넓어져 경계가 단순·부드러워짐 → **과소적합 경향**.

---

### 2. 트리 및 배깅 계열

#### Decision Tree
* **`max_depth` (최대 트리 깊이):**
  * **증가 시:** 분기를 깊게 이어가며 **과적합 위험 상승**.
  * **감소 시:** 트리가 단순해져 **과소적합 위험 상승**, 해석·일반화 유리.
* **`min_samples_split` / `min_samples_leaf`:** 분할/말단 노드에 필요한 최소 샘플 수.
  * **증가 시:** 세밀한 말단 분기를 제한해 **과적합 억제**.
* **`criterion` (`gini`, `entropy`, `log_loss`):** 분할 기준. 불순도 계산 방식에 따른 미세 분기 차이.

#### Bagging Classifier
* **`n_estimators` (기본 추정기 수):** 증가 시 평균/투표 안정성이 높아져 **분산 감소**, 대신 학습·예측 시간 증가.
* **`max_samples` / `max_features`:** 각 기본 모델이 사용할 샘플/특성 비율. 낮출수록 모델 간 다양성 증가(너무 낮으면 개별 모델 약화로 과소적합 가능).
* **`estimator`:** 기본 모델(보통 Decision Tree). 기반 모델의 복잡도가 전체 앙상블 특성에 직접 반영됨.

#### Random Forest
* **`n_estimators`:** 트리 수. 증가 시 예측 분산 감소 및 안정화, 연산 시간 증가.
* **`max_features`:** 각 분기에서 고려할 특성 수. 줄일수록 트리 간 상관관계가 낮아져 분산 감소에 유리하나, 너무 작으면 핵심 변수 선택이 어려울 수 있음.
* **`max_depth`, `min_samples_split`, `min_samples_leaf`:** 개별 트리 크기 제한을 통한 과적합 제어 및 속도 조절.

#### Extra Trees
* **`n_estimators`:** 트리 수. 증가 시 분산 감소, 계산 비용 증가.
* **`max_features`:** 분기 후보 특성 수. 작을수록 트리 간 다양성 증가.
* **`max_depth`, `min_samples_split`:** 분할 임계값을 무작위로 고르는 특성상 트리가 깊어질 수 있으므로 깊이/샘플 제한으로 과적합과 속도를 조절.

---

### 3. 트리 · 부스팅 계열

#### AdaBoost
* **`n_estimators` (약한 학습기 수):** 증가 시 틀린 샘플을 교정하는 단계가 늘어나지만, 과도하면 **과적합** 및 노이즈 샘플 과집중 위험.
* **`learning_rate`:** 각 약한 학습기의 반영 비율. 낮출수록 천천히 학습해 일반화에 유리하지만, 충분한 성능을 내려면 `n_estimators`를 더 늘려야 함.

#### Gradient Boosting
* **`n_estimators` (트리 수):** 증가 시 잔차/기울기를 줄이는 반복이 늘어나지만, 과도하면 **과적합** 발생.
* **`learning_rate`:** 각 트리의 기여 축소 비율. 낮출수록 과적합 억제는 유리하나 더 많은 트리(`n_estimators`)가 필요.
* **`max_depth` / `subsample`:** `max_depth`가 크면 복잡한 상호작용을 잡지만 과적합 쉬움. `subsample`을 `1.0` 미만으로 두면 일부 샘플만 써서 과적합 완화와 속도 향상에 도움.

---

### 4. 신경망 계열

#### Perceptron
* **`learning_rate` (`eta0`):** 크면 초기 변화가 빠르지만 진동/발산 가능, 작으면 느리지만 안정적.
* **`max_iter` / `tol`:** 최대 반복 횟수 및 수렴 판정 기준. 선형 분리가 어려운 데이터에서는 최대 반복까지 도달할 수 있음.

#### Multi-Layer Perceptron (MLP)
* **`hidden_layer_sizes`:** 은닉층 수와 유닛 수. 늘릴수록 비선형 표현력이 커지지만, 데이터 대비 과도하면 **과적합**과 학습 시간 증가.
* **`activation` (`relu`, `tanh`, `logistic`):** `relu`는 빠른 학습과 기울기 소실 완화에 유리, `tanh`/`logistic`은 포화 구간에서의 기울기 소실 주의.
* **`alpha` ($L_2$ 정규화 강도):** 클수록 가중치 크기를 강하게 제약해 **과적합 억제**.
* **`learning_rate_init` / `batch_size` / `max_iter`:** 초기 학습률, 미니배치 크기, 최대 에폭 수에 따라 수렴 속도와 일반화 차이가 발생.

#### Deep Neural Network (DNN)
* **은닉층 깊이(Depth) 및 폭(Width):** 깊고 넓을수록 표현력이 커지지만 **과적합, 기울기 소실/폭발, 연산량** 증가 위험도 커짐.
* **학습률 및 옵티마이저 (`Adam`, `SGD` 등):** 학습률이 너무 크면 발산/진동, 너무 작으면 수렴 지연.
* **정규화·안정화 (`Dropout` 비율, `BatchNorm`, `Weight Decay`):**
  * `Dropout` 비율이 높으면 과적합 억제는 유리하나 너무 높으면 과소적합 가능.
  * `BatchNorm`과 `Weight Decay`는 학습 안정화와 가중치 크기 제어에 도움.
