# 화재 재난 상황 인식을 위한 객체 검출
### Object Detection for Fire Disaster Situation Recognition

> AOK 2022 학술발표대회 논문집 (29권 2호), pp.426–428

**저자**: 김태성¹ · 방재연² · 서정운² · 손경아¹,³
㈜ ¹아주대학교 인공지능학과 · ²아주대학교 심리학과 · ³아주대학교 소프트웨어학과

---

## 📌 개요

화재 상황에서의 빠른 현장 파악은 인명 피해를 줄이는 데 중요한 요소입니다. 기존 연구는 화재와 관련된 데이터셋들은 대부분 불과 연기를 라벨링하여 화재의 여부에만 초점을 두고 있는 한계가 있습니다.

본 연구에서는 화재 상황에서 **사람과 소방관, 연기, 불**을 탐지하는 Object Detection 모델을 만들어 현장 파악에 도움을 주고자 하였습니다. 이를 위해 화재 상황 이미지 약 3000장을 수집·라벨링하여 데이터셋을 구성하였고, 이를 이용해 객체 검출 모델인 **RetinaNet**을 학습하였습니다.

또한 화재 상황에서 Object Detection 모델의 성능을 향상시키기 위해, 기존 RetinaNet에 **Dehazing(FFA-Net)**, **Smoke Augmentation**, **Semi-supervised(ISD)** 방법을 적용하여 각각의 조건에서 mAP 63.7로 가장 높은 성능을 도출하였습니다.

---

## 1. 서론

2011년부터 2020년 사이 매년 평균 42,332건의 화재가 발생하였으며, 화재로 인한 2216명의 인명피해와 309명의 사상자가 발생했습니다 [1]. 화재 재난 상황을 빠르게 파악하여 인명구조에 도움을 주기 위해 대상의 위치와 손실성을 탐지하는 object detection을 활용할 수 있습니다.

기존 화재와 관련된 object detection 연구들은 대부분 object의 class를 **fire(불)**과 **smoke(연기)**로 한정하여 화재 상황 대응에는 불충분합니다. 이에 본 연구는 화재 이미지 데이터를 수집하고, 객체 탐지 대상을 **불, 연기, 사람, 소방관**으로 설정하여 연구를 진행하였습니다.

또한 많은 화재 상황 이미지가 진하고 어두운 연기로 인해 object detection의 성능이 저하될 수 있어, 이에 대처하기 위해 **smoke augmentation**과 **dehazing** 방법을 사용하였습니다. 라벨링 데이터를 필요로 하는 supervised 학습을 사용하기 위해서는 라벨링에 소요되는 시간이 많이 필요한데, 라벨링되지 않은 데이터도 최대한 활용하기 위해 **semi-supervised** 방법을 사용하였습니다.

---

## 2. 데이터셋

### 2.1 화재 상황 베이스셋

FireNET [4], Domestic-Fire-Smoke-Dataset [5] 데이터와 여러 화재 상황의 비디오 콘텐츠를 활용해 python의 pytube, cv2, selenium 라이브러리를 이용하여 화재 동영상을 크롤링하고, 프레임을 잘라 이미지를 수집하였습니다. 이후 베이스셋의 품질을 위해 라벨링과 이미지를 검토하는 과정을 거쳐 총 3007장의 이미지를 최종 labeled 베이스셋으로 선정하였고, 이 중 2675장을 trainset, 332장을 testset으로 분할하였습니다.

Semi-supervised 기법을 위한 unlabeled 데이터의 경우 labeled 베이스셋에 쓰이지 않은 YouTube 화재 동영상을 사용하였습니다. 추가로 smoke augmentation을 위해 smoke 키워드로 구글 이미지 [6]에서 연기 이미지를 크롤링하였습니다.

### 2.2 라벨링

labelImg [7] 프로그램을 활용하여 3007장의 화재 상황 이미지를 **fire, smoke, person, firefighter** 4개의 class에 대해 라벨링하였습니다.

| | Labeled | Unlabeled | Total |
|---|---|---|---|
| 이미지 개수 | 3007 | 1154 | 4161 |

| | Trainset | Testset | Total Labeled |
|---|---|---|---|
| 이미지 개수 | 2675 | 332 | 3007 |

| Class | Trainset | Testset | Total |
|---|---|---|---|
| Fire | 2986 | 328 | 3314 |
| Smoke | 2675 | 375 | 3050 |
| Person | 926 | 84 | 1010 |
| Firefighter | 2437 | 419 | 2856 |

---

## 3. 실험 조건

본 연구에서 사용한 object detection 모델은 **RetinaNet** [8]입니다. RetinaNet은 one-stage detector로 속도가 빠르고 여러 스케일의 object를 효율적으로 탐지하기 위한 FPN(Feature Pyramid Network)과 클래스 불균형 문제를 다루기 위한 Focal loss를 적용한 것이 주요 특징입니다.

### 3.1 Image dehazing

Image dehazing은 안개, 연기, 먼지 등이 있는 이미지에서 선명한 이미지로 복원하는 기술입니다. 화재 상황에서 object detection 성능에 연기가 미치는 영향을 줄이기 위해 전처리 방법으로 사용하였습니다. Dehazing 모델 중 **FFA-Net** [9]을 사용하였습니다. FFA-Net은 CNN 기반의 dehazing network로, 픽셀과 채널마다 다른 가중치를 주는 모듈을 추가해 중요한 정보를 가진 픽셀과, 채널에 집중하여 성능을 개선하였습니다.

### 3.2 Smoke augmentation

선행 논문인 NSA [10], FPI [11]에서 사용했던 기법을 object detection에 맞게 수정하여, 원본 이미지에 연기를 더해 모델을 학습시키고자 하였습니다.

과정은 아래와 같습니다.
1. 목표 이미지의 다빈도를 확인, 연기가 있는지 확인한다.
2. 두 과목의 선정된 연기 이미지로부터 일정 크기의 패치를 잘라낸다.
3. 패치의 크기를 변환한 후, 패치를 목표 이미지의 임의의 위치에 더한다.
4. 패치가 더해진 이미지에 대하여 Poisson blending [12]을 실시한다.
5. 추가된 연기 patch data를 라벨에 추가한다.

이 조건에서, 크롤링을 통해 얻어진 121개의 연기 이미지를 smoke augmentation에 활용하였습니다.

### 3.3 Semi-supervised Learning (ISD)

semi-supervised learning은 labeled 데이터와 unlabeled 데이터 모두를 사용해서, 모델을 학습시키는 방법입니다. 본 연구는 그 중, 연기의 시야 차폐가 mix-up [13] 기법과 유사하여, mix-up을 활용한 semi-supervised인 **ISD** [14] (Interpolation-based Semi-supervised Learning for Object Detection) loss를 사용하였습니다.

mix-up은 이미지 둘을 합이 1이되는 서로 다른 가중치를 곱한 뒤, 더하는 방식입니다. ISD loss는 mix-up한 이미지와 원본 이미지 둘에 대한 모델의 예측을 비교하여 loss를 얻는 방식으로, object와 background인 경우를 나누어서 loss함수를 구현합니다.

ISD loss의 성능을 평가하기 위해서 unlabeled 데이터 1154개를 추가한 조건과(w/), 추가하지 않은 조건(w/o)을 따로 비교하였습니다.

### 3.4 실험 설정

각 조건을 RetinaNet에 적용하여 실험을 실시하였습니다. epoch은 100으로 설정하였고, **ResNet-50 FPN** 네트워크를 backbone으로 사용하였습니다. 평가지표로는 **AP(Average Precision)**를 사용하였으며, **IoU threshold는 50%**로 설정하였습니다.

---

## 4. 실험 결과

### 4.1 Smoke Augmentation, Dehazing 조건 실험 결과

| Class | RetinaNet 원본 | Smoke Augmentation | Dehazing (FFA) |
|---|---|---|---|
| Fire | 53.7 | **56.2** | 51.7 |
| Smoke | 70.4 | 72.6 | 65.5 |
| Person | **52.5** | 52.2 | 38.7 |
| Firefighter | 70.9 | **71.9** | 67.2 |
| **mAP** | 61.8 | **63.2** | 55.7 |

원본 조건과 비교했을 때 smoke augmentation과 FFA-Net을 활용한 Dehazing 조건의 클래스별 AP와 평균 AP(mAP)를 비교한 결과, smoke augmentation을 적용한 조건이 가장 높았고, person의 class AP만 원본 RetinaNet이 가장 높았습니다.

### 4.2 Semi-supervised 조건 실험 결과

| Class | RetinaNet 원본 | ISD (w/o) | ISD (w/) |
|---|---|---|---|
| Fire | 53.7 | 55.5 | **59** |
| Smoke | 70.4 | 72.5 | **75.2** |
| Person | 52.5 | 44.0 | 47.6 |
| Firefighter | 70.9 | 70.6 | **73.1** |
| **mAP** | 61.8 | 60.6 | **63.7** |

원본 조건과 비교해서, semi-supervised loss인 ISD loss를 적용한 조건을 비교하였습니다. w/o 조건은 unlabeled 데이터를 추가하지 않은 조건이고, w/ 조건은 unlabeled 데이터 1154장을 추가한 조건입니다. ISD loss를 추가했을 때, 원본과 비교하여 person의 class AP가 많이 떨어졌지만, fire와 smoke의 class AP는 높아졌습니다. ISD의 w/ 조건과 w/o 조건을 비교하였을 때, w/ 조건에서 모든 class의 AP가 높아진 것을 볼 수 있었습니다.

종합적으로, 가장 mAP가 높은 조건은 unlabeled 데이터를 추가한 ISD 조건이며, person class의 AP가 크게 차이 나지 않고 mAP가 높아진 조건은 **Smoke augmentation**이었습니다.

---

## 5. 결론

본 연구는 화재 상황에서의 object detection 적용을 위해, 불(fire), 연기(smoke), 사람(person), 소방관(firefighter)으로 라벨링한 화재 상황 데이터셋을 구축하였습니다. 또한 Smoke augmentation, dehazing(FFA), Semi-supervised Learning(ISD)을 RetinaNet에 적용·비교 분석하였습니다. 그 중 Semi-supervised Learning(ISD) 조건의 mAP가 가장 높았지만 person class의 정확도가 떨어지는 단점을 보였습니다. 추후 연구를 통해 제안된 모델들의 person class AP를 향상시키고, 기존 화재 예방을 목적으로 한 모델들과 통합할 예정입니다.

---

## 사사

본 연구는 과학기술정보통신부 및 정보통신기획평가원의 대학ICT연구센터육성지원사업의 연구결과로 수행되었음 (IITP-2022-2018-0-01431)

---

## 참고문헌

1. e나라지표, https://www.index.go.kr/potal/main/EachDtlPageDetail.do?idx_cd=1632
2. Wu, Shixiao, and Libing Zhang. "Using popular object detection methods for real time forest fire detection." 2018 11th International symposium on computational intelligence and design (ISCID). Hangzhou, China. 2018. Vol. 1, pp. 280-284.
3. Li, Pu, and Wangda Zhao. "Image fire detection algorithms based on convolutional neural networks." Case Studies in Thermal Engineering. Vol.19. Article. 100625. 2020.
4. OlafenwaMosesRoger/FireNET, 2019. https://github.com/OlafenwaMoses/FireNET
5. datacluster-labs/Domestic-Fire-and-Smoke-Dataset, 2021. https://github.com/datacluster-labs/Domestic-Fire-and-Smoke-Dataset
6. Google Images, https://www.google.co.kr/imghp?hl=ko
7. heartexlabs/labelImg, 2022. https://github.com/heartexlabs/labelImg
8. Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. Venice, Italy. 2017. pp. 2980-2988.
9. Qin, Xu, et al. "FFA-Net: Feature fusion attention network for single image dehazing." Proceedings of the AAAI Conference on Artificial Intelligence. New York, USA. 2020. Vol. 34, No. 07. pp. 11908-11915.
10. Schlüter, Hannah M., et al. "Self-supervised out-of-distribution detection and localization with natural synthetic anomalies (nsa)." arXiv preprint arXiv:2109.15222. 2021.
11. Castro, Eduardo, Jaime S. Cardoso, and Jose Costa Pereira. "Elastic deformations for data augmentation in breast cancer mass detection." 2018 IEEE EMBS International Conference on Biomedical & Health Informatics (BHI). Nevada, USA. 2018. pp. 230-234.
12. P. Ferez, M. Gangnet, and A. Blake. "Poisson image editing." ACM Transactions on graphics (TOG). Vol.22. pp. 313-318. 2003.
13. Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412. 2017.
14. Jeong, Jisoo, et al. "Interpolation-based semi-supervised learning for object detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. Tennessee, USA. 2021. pp.11602-11611.

---

## Citation

논문을 인용하실 경우 아래 형식을 참고해 주세요.

```
김태성, 방재연, 서정운, 손경아. "화재 재난 상황 인식을 위한 객체 검출."
AOK 2022 학술발표대회 논문집 29.2 (2022): 426-428.
```
