# RNN · LSTM · GRU 핵심 개념 정리

> 순차 데이터(sequential data) 처리를 위한 순환 신경망 계열 모델을 기술면접 대비용으로 정리한 학습 노트입니다.

## 목차

- [1. RNN과 장기 의존성 문제](#1-rnn과-장기-의존성-문제)
- [2. LSTM](#2-lstm)
- [3. GRU](#3-gru)
- [4. LSTM vs GRU 게이트 대응 정리](#4-lstm-vs-gru-게이트-대응-정리)
- [부록. 자주 헷갈리는 용어](#부록-자주-헷갈리는-용어)

---

## 1. RNN과 장기 의존성 문제

### RNN의 기본 동작

RNN은 순차 데이터를 처리하기 위해 **hidden state(hₜ)**를 사용한다. 매 시점 입력 xₜ와 이전 hidden state hₜ₋₁을 함께 받아 현재 hidden state를 갱신하며, 과거 정보를 hidden state에 담아 다음 시점으로 전달한다.

### 문제 — 장기 의존성(Long-term Dependency) 문제

시퀀스가 길어질수록 RNN은 **먼 과거의 정보를 학습에 반영하기 어려워진다**. 이를 **장기 의존성 문제(Long-term Dependency Problem)**라 부른다.

### 근본 원인 — BPTT에서의 Vanishing Gradient

RNN은 시간 축을 따라 펼친 뒤 역전파하는 **BPTT(Backpropagation Through Time)**로 학습한다. 이때 gradient가 과거로 거슬러 올라가려면 체인룰에 의해 ∂hₜ/∂hₜ₋₁ 항이 **시점 수만큼 반복해서 곱해진다.**

이 항은 대략 **tanh'(·) × W_hh** 형태인데,

- tanh의 도함수는 항상 **0~1 사이**의 값을 가지므로,
- 이를 수십~수백 번 곱하면 gradient가 **지수적으로 0에 수렴**(소실)한다.

→ 그 결과 앞쪽(먼 과거) 시점일수록 gradient가 거의 전달되지 않아 학습이 제대로 이루어지지 않는다.

> ⚠️ 표현 주의: "미분에 참여하는 시퀀스가 많아진다"보다는 **"체인룰이 시점 수만큼 반복 적용되며 gradient가 누적적으로 곱해진다"**가 더 엄밀한 표현이다.

---

## 2. LSTM

LSTM(Long Short-Term Memory)은 RNN의 장기 의존성 문제를 완화하기 위해 설계됐다.

### 2-1. RNN과의 가장 큰 구조적 차이 — Cell State

RNN에는 hidden state 하나뿐이지만, LSTM에는 두 종류의 상태가 **분리**되어 존재한다.

| 상태 | 의미 |
| --- | --- |
| **Hidden state (hₜ)** | 단기 정보(short-term) |
| **Cell state (cₜ)** | 장기 정보(long-term) — 장기 의존성 완화의 핵심 |

> "게이트가 있다"는 절반짜리 답이다. 가장 큰 차이는 **Cell state의 도입**이다.

### 2-2. 세 개의 게이트

| 게이트 | 역할 |
| --- | --- |
| **Forget gate (fₜ)** | 이전 cell state(cₜ₋₁)에서 **불필요한 과거 정보를 얼마나 잊을지** 조절 |
| **Input gate (iₜ)** | 현재 시점의 새로운 정보(임시 cell state c̃ₜ)를 **얼마나 반영할지** 조절 |
| **Output gate (oₜ)** | 갱신된 cell state를 **얼마나 hidden state로 내보낼지** 조절 |

> ⚠️ Forget gate와 Input gate는 역할이 명확히 구분된다(잊기 vs 새로 받기). 둘을 묶어 "입력 반영을 조율한다"고만 하면 구분이 희석된다.

### 2-3. 동작 흐름

1. Forget gate가 이전 cell state에서 버릴 정보를 정하고,
2. Input gate가 새 후보 정보 c̃ₜ를 얼마나 더할지 정해
3. **Cell state를 갱신**: cₜ = fₜ ⊗ cₜ₋₁ ⊕ iₜ ⊗ c̃ₜ
4. Output gate가 갱신된 cell state를 반영해 **hidden state hₜ를 계산**

(⊗: element-wise 곱, ⊕: element-wise 덧셈)

### 2-4. 왜 장기 의존성이 완화되는가

Cell state는 위처럼 **element-wise 덧셈**으로 갱신된다. 따라서 역전파 시 gradient가 시간을 거슬러 올라갈 때 **곱셈이 연속으로 반복되지 않고** 비교적 직접적으로 흐를 수 있어 Vanishing Gradient가 완화된다. 또한 Forget gate가 **필요한 과거 정보만 선택적으로 유지**할 수 있다.

### 2-5. (보충) Peephole Connection

기본 LSTM에서 각 게이트는 입력 xₜ와 hidden state hₜ₋₁만 보고 값을 정한다. **Peephole connection**은 여기에 더해 게이트가 **cell state(cₜ₋₁ 또는 cₜ)까지 직접 들여다보도록(peep)** 연결을 추가한 변형이다. 이를 통해 게이트가 셀의 현재 상태를 참고해 타이밍을 더 정밀하게 제어할 수 있다.

---

## 3. GRU

GRU(Gated Recurrent Unit)는 **LSTM을 구조적으로 단순화**한 모델이다.

### 3-1. 핵심 단순화

1. **게이트 통합:** LSTM의 Forget gate + Input gate가 하나의 **Update gate(zₜ)**로 통합됐다.
2. **상태 통합:** LSTM의 **Cell state와 Hidden state가 하나의 Hidden state로 통합**됐다.

→ 그 결과 **파라미터 수가 줄어 LSTM보다 효율적**이다. (게이트 통합과 상태 통합 모두가 파라미터 감소의 이유다.)

### 3-2. 두 개의 게이트

| 게이트 | 역할 |
| --- | --- |
| **Update gate (zₜ)** | 이전 hidden state를 얼마나 유지하고 새 정보를 얼마나 반영할지 조절. 수식상 **(1 − zₜ)가 forget 역할, zₜ가 input 역할**을 동시에 수행 |
| **Reset gate (rₜ)** | 임시 hidden state(h̃ₜ)를 계산할 때 **이전 hidden state(hₜ₋₁)를 얼마나 반영할지** 조절 |

> ⚠️ 용어 주의: Reset gate는 "hidden state로 도입된" 것이 아니라 **게이트**다. state와 gate를 혼동하지 말 것. (강의 자료 기준으로는 Reset gate를 LSTM의 Output gate에 대응시켜 설명한다.)

---

## 4. LSTM vs GRU 게이트 대응 정리

| 구분 | LSTM | GRU |
| --- | --- | --- |
| 상태 | Hidden state + Cell state (분리) | Hidden state 하나로 통합 |
| 정보 잊기/받기 | Forget gate + Input gate | **Update gate (zₜ)** 하나로 통합 |
| 과거 반영 조절 | Output gate | **Reset gate (rₜ)** |
| 파라미터 수 | 많음 | 적음 (더 효율적) |

---

## 부록. 자주 헷갈리는 용어

| 구분 | 올바른 설명 |
| --- | --- |
| **State vs Gate** | State(hₜ, cₜ)는 정보를 담아 전달하는 값. Gate(fₜ, iₜ, oₜ, zₜ, rₜ)는 정보 흐름의 양을 0~1로 조절하는 장치 |
| **Long-term Dependency** | 시퀀스가 길 때 먼 과거 정보를 학습에 반영하기 어려운 문제. 원인은 BPTT에서의 Vanishing Gradient |
| **BPTT** | 시간 축으로 펼친 RNN에 대한 역전파(Backpropagation Through Time) |
| **Cell state 덧셈 갱신** | LSTM이 gradient 소실을 완화하는 핵심. 곱셈 누적이 아닌 덧셈으로 정보가 흐름 |

---

*본 노트는 RNN 계열 모델 학습 내용을 기술면접 대비 관점에서 재정리한 것입니다.*
