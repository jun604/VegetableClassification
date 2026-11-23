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
## 성능 비교: 각 모델별 정확도가 가장 높은 5개의 파라미터 조합을 가져와서 평균값 비교
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
