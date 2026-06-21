# CNN 아키텍처 핵심 개념 정리

> 딥러닝 CNN 대표 아키텍처(VGGNet · InceptionNet · ResNet)와 관련 핵심 개념을 기술면접 대비용으로 정리한 학습 노트입니다.

## 목차

- [1. CNN 기본 구조](#1-cnn-기본-구조)
- [2. VGGNet](#2-vggnet)
- [3. InceptionNet (GoogLeNet)](#3-inceptionnet-googlenet)
- [4. ResNet](#4-resnet)
- [5. Global Average Pooling (GAP)](#5-global-average-pooling-gap)
- [6. VGGNet vs ResNet 구조 비교](#6-vggnet-vs-resnet-구조-비교)
- [부록. 자주 헷갈리는 용어](#부록-자주-헷갈리는-용어)

---

## 1. CNN 기본 구조

일반적인 CNN은 크게 두 파트로 나뉜다.

| 파트 | 구성 레이어 | 역할 |
| --- | --- | --- |
| **Feature Extractor** | Convolution + Pooling | 입력 이미지의 공간·위치 정보를 고려하며 각 영역의 고수준 특징을 추출 |
| **Classifier** | Fully-Connected (FC) Layer | 추출된 특징을 종합해 입력 이미지가 무엇인지 최종 분류 |

> 두 파트의 명칭(**Feature Extractor / Classifier**)은 반드시 정확히 기억할 것.

**학습되는 파라미터**

- Feature Extractor: 각 **필터(커널)의 가중치**와 편향(bias)
- Classifier: 각 노드 간 **연결 가중치(weight)**와 편향(bias)

Pooling은 학습 파라미터가 없는 연산이라는 점에 유의한다(공간 크기만 줄임).

---

## 2. VGGNet

### 핵심 아이디어 — 작은 Conv를 깊게 쌓기

VGGNet은 큰 필터를 한 번 쓰는 대신 **3×3 Convolution을 여러 번 중첩(stack)**한다.

### Receptive Field 관점

3×3 Conv를 두 번 중첩하면 **5×5 Conv 한 번과 동일한 Receptive Field**(5×5 영역)를 바라본다.

### 5×5 단일 Conv 대비 장점

입력 채널 수를 C, 필터(출력 채널) 수를 K라 하면:

| 방식 | 파라미터 수 |
| --- | --- |
| 5×5 단일 Conv | 5 × 5 × C × K = **25CK** |
| 3×3 Conv 두 번 | 2 × (3 × 3 × C × K) = **18CK** |

→ 같은 Receptive Field를 보면서도 파라미터가 **25CK → 18CK**로 줄어 더 효율적이다.

> ⚠️ 흔한 실수: "18개 vs 25개"처럼 채널 수(C)와 필터 수(K)를 빼고 절댓값으로만 비교하면 안 된다. 정확한 비교는 **18CK vs 25CK**이다.

> 보충: 중첩 시 중간 영역이 여러 번 탐색(overlapping)되어 더 풍부한 특징을 뽑아낼 수 있다는 장점도 있다.

### 사이즈 축소 방식

- Conv Layer는 사이즈를 **유지**(padding=same)
- 사이즈 축소는 오직 **Pooling Layer에서만** 발생
- 마지막 Pooling: **Max Pooling**

---

## 3. InceptionNet (GoogLeNet)

### 3-1. 1×1 Convolution

**목적:** 연산 부담 감소.

**원리:** 1×1 Conv는 입력의 가로·세로(공간 해상도)는 그대로 유지하면서 필터 개수로 출력 **채널 수(depth)**를 자유롭게 조절한다. 이렇게 채널 수를 줄이면, 뒤따르는 3×3·5×5 Conv의 **입력 채널 수가 줄어 그 Conv들의 파라미터 수가 감소**한다.

> ⚠️ 용어 주의: 줄이는 대상은 "층(layer)"이 아니라 feature map의 **채널 수(channel, depth)**이다.

### 3-2. Inception Module

여러 크기의 필터를 **병렬로 동시에** 사용한다.

| 브랜치 | 구성 |
| --- | --- |
| 1 | 1×1 Conv |
| 2 | 1×1 Conv → 3×3 Conv |
| 3 | 1×1 Conv → 5×5 Conv |
| 4 | 3×3 Max Pooling → 1×1 Conv |

**병렬 사용 이유:** 서로 다른 크기의 필터는 서로 다른 스케일의 특징(feature)을 추출한다. 어떤 크기가 최적인지 미리 알 수 없으므로 여러 스케일의 특징을 동시에 뽑아내기 위함이다.

**출력 결합 방법:** 각 브랜치의 출력 공간 크기가 동일하도록 **padding을 사전에 설정**한 뒤(아래 표), **채널(깊이, Z축) 방향으로 Concatenation(연결)**한다.

| 필터 | Padding |
| --- | --- |
| 1×1 Conv | 0 |
| 3×3 Conv | 1 |
| 5×5 Conv | 2 |
| 3×3 Max Pooling | 1 |

> Padding은 Convolution 이전에 필터 크기·stride·채널 수와 함께 설정하는 하이퍼파라미터다. 흐름: **패딩 사전 설정 → 패딩 적용 → Conv 연산 → 출력**.

### 3-3. Pooling 위치 변경

4번째 브랜치에서 Max Pooling과 1×1 Conv의 **순서를 바꾼** 이유:

Inception Module이 여러 개 쌓이는 구조에서 Pooling이 모듈 경계에 위치하면 **모듈 경계마다 Pooling이 연속으로 발생**하는 구간이 생긴다. Pooling은 정보를 압축하는 연산이라, 빈번히 반복되면 **정보 손실이 과도**해진다. 이를 막기 위해 위치를 변경했다.

---

## 4. ResNet

### 4-1. 해결하려는 문제

BN·ReLU 조합으로 Vanishing Gradient는 해결했지만, **모델이 너무 깊어지면 오히려 학습이 안 되는** 문제가 발견됐다(강의 노트 표현: *오히려 underfitting*). 이후 밝혀지길, 깊어질수록 **Loss 표면(landscape)이 매우 울퉁불퉁(꼬불꼬불)해져** 수렴이 어려워진다.

> 참고: ResNet 원 논문은 이를 **Degradation Problem**(층을 깊게 쌓을수록 train error조차 나빠지는 현상)이라 명명한다. 강의 기준으로는 "underfitting", 면접 맥락에서는 두 용어 모두 알아두면 좋다.

### 4-2. Skip Connection (Residual Connection)

**핵심 아이디어:** 입력값 x를 더 깊은 층으로 **직접 그대로 전달**한다.

입력을 x, 블록의 출력을 F(x), 목표(정답) 매핑을 H(x)라 하면:

```
H(x) = F(x) + x
F(x) = H(x) - x
```

이상적인 매핑이 항등(x)에 가깝다면 **F(x)는 0에 가까워지면** 된다. 깊은 층일수록 파라미터를 0에 가깝게 만드는 것이, 특정 값에 정확히 맞추는 것보다 쉽기 때문에 학습이 수월해진다.

> 즉, Skip Connection에서 모델이 실제로 학습하는 것은 **잔차(Residual) F(x) = H(x) − x** 이다.

### 4-3. Bottleneck 구조

**적용 기준:** 약 **50층 이상**의 깊은 모델.

**구조:** **1×1 Conv → 3×3 Conv → 1×1 Conv**

1. 1×1 Conv로 채널 수를 **줄이고**
2. 줄어든 채널에서 3×3 Conv 연산(파라미터 절감)
3. 1×1 Conv로 채널 수를 **원상복구**

**채널 복원 이유:** Skip Connection으로 들어오는 입력과 **채널 수가 같아야 element-wise(행렬) 덧셈**이 가능하기 때문.

채널이 줄었다가 다시 늘어나는 모양이 병목과 닮았다고 하여 **Bottleneck**이라 부른다.

### 4-4. ResNet v1 vs v1.5 — Downsampling

| 버전 | Downsampling 방식 | 문제/개선 |
| --- | --- | --- |
| **v1** | 1×1 Conv에 stride=2 | 1×1 + stride=2는 건너뛰는 영역이 생겨 **입력의 절반 정도가 연산에 참여하지 못함 → 정보 손실** |
| **v1.5** | 3×3 Conv에 stride=2 | 필터 크기(3)가 stride(2)보다 커서 인접 영역이 **겹치며(overlapping) 모든 입력 영역을 빠짐없이 커버** |

---

## 5. Global Average Pooling (GAP)

InceptionNet 이후 ResNet 등 대부분의 모델은 Feature Extractor와 Classifier 사이에서 **Flatten 대신 GAP**를 쓴다.

**Flatten 방식의 문제:** 3D feature map을 FC Layer에 넣으려고 1D 벡터로 펴면, **FC Layer가 전체 파라미터에서 차지하는 비중이 압도적으로 커져** 모델이 비효율적이고 과적합(overfitting)되기 쉽다.

**GAP의 작동 방식:** 각 **채널마다** feature map의 모든 공간 위치 **활성화 값(activation value)을 하나의 평균값으로** 압축한다.

```
예: 7×7×512  ──GAP──▶  1×1×512
```

> ⚠️ 용어 주의: 평균을 내는 대상은 학습되는 "파라미터"가 아니라 feature map의 **활성화 값**이다. 또한 "층"이 아니라 **채널**별로 평균을 낸다.

**왜 한 값으로 압축해도 괜찮은가:** 깊은 Conv를 거치며 픽셀 하나가 거의 이미지 전체를 포괄하는 **고수준의 유의미한 특징**을 담게 되므로, 공간을 평균내도 정보가 충분히 보존된다. GAP를 쓰면 파라미터가 급감해 **과적합을 막는** 효과가 있고, 이후 FC 한 층만 통과시켜 Conv의 효과를 최대로 활용한다.

---

## 6. VGGNet vs ResNet 구조 비교

| 항목 | VGGNet | ResNet |
| --- | --- | --- |
| 사이즈 축소 방식 | **Pooling Layer에서만** 축소 (Conv는 padding=same으로 유지) | 중간 축소는 **Conv의 stride=2**로 수행 |
| Pooling 개수/위치 | 여러 개의 Max Pooling | 맨 앞 **Max Pooling**, 맨 뒤 **GAP** 총 2개 |
| 마지막 Pooling 종류 | **Max Pooling** | **Global Average Pooling (GAP)** |
| 핵심 구조 | 3×3 Conv 중첩 | Skip Connection (Residual) |

---

## 부록. 자주 헷갈리는 용어

| 구분 | 올바른 설명 |
| --- | --- |
| **파라미터(parameter)** | 모델이 학습으로 업데이트하는 값 — Conv 필터 가중치/bias, FC weight/bias, BatchNorm의 γ·β 등 |
| **활성화 값(activation)** | feature map에 담긴 출력 값. 학습 대상이 아님 (GAP가 평균내는 대상) |
| **채널(channel, depth)** | feature map의 깊이. "층(layer)"과 혼동 금지 |
| **Pooling** | 학습 파라미터 없음. **공간 크기(가로×세로)**를 줄임 (파라미터 수를 줄이는 것이 아님) |
| **Receptive Field** | 출력의 한 픽셀이 입력에서 바라보는 영역의 크기 |

---

*본 노트는 CNN 아키텍처 학습 내용을 기술면접 대비 관점에서 재정리한 것입니다.*
