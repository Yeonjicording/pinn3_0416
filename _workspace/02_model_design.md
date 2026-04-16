# 02. 모델 아키텍처 설계 — Inverse-PINN for Scout Odometry Correction

## 문제 정의
- 문제 유형: 회귀 + 물리제약 역문제 (Inverse Problem)
- 입력 형상: `(B, 6)` — standardized `[d_x, d_y, d_z, d_rolling, d_pitch, d_yaw]` (odom)
- 부가 입력: `x_raw (B,6)` 원단위 odom, `stationary_mask (B,)` bool
- 출력 형상: `(B, 2)` — `[d_x_corr, d_yaw_corr]` (원단위, m / rad)
- Learnable system coefficients: `b` (baseline), `s_r` (wheel radius scale), `α_sum = α_L + α_R` (좌우 휠 평균 스케일 — body-frame delta만 주어지므로 α_L, α_R 개별 식별 불가, 평균 항만 관측 가능)
- 평가 메트릭: MSE, ATE, RPE, FPE (data_module의 `accumulate_trajectory` 이용)

## 설계 결정 요약

### Inverse-PINN 구조 — 옵션 (A) 채택
- **(A) MLP 직접 보정 + 별도 Parameter로 계수**: MLP는 `[d_x_corr, d_yaw_corr]` 를 직접 예측하고, 학습 가능한 스칼라 파라미터 `b, s_r, α_L, α_R`는 Physics loss 항(특히 non-holonomic residual과 coefficient prior)에서 역추정 역할을 함.
- **(B) MLP residual + 휠 모델 결합**은 raw 좌우 휠 속도 `(v_L, v_R)`가 필요한데 본 데이터는 body-frame delta만 제공하므로 한계 있음. 따라서 채택하지 않음.
- 선정 근거: 데이터 접근성(body-frame delta only), MLP의 표현력, 그리고 계수의 해석 가능성을 동시에 확보.

### Activation 선정 — SiLU (Swish)
- PINN 관례상 `tanh`/`SiLU`/`GELU` 같은 smooth activation이 권장됨. 이유: physics residual이 고차 미분 또는 연속성에 의존할 수 있고, ReLU의 구간 선형성은 gradient 흐름에서 sharp edge를 만들어 물리제약 최적화를 불안정하게 함.
- 본 과제는 2차 미분은 쓰지 않지만, coefficient prior와 non-holonomic residual의 연속성 덕분에 SiLU가 안정적. tanh 대비 saturation이 덜해 standardized 입력에서 수렴 속도 유리.

## 아키텍처 다이어그램 (텍스트)

```
       x (B,6) standardized odom delta
            │
     ┌──────▼──────┐
     │ Linear 6→H  │
     │   SiLU      │
     ├─────────────┤
     │ Linear H→H  │
     │   SiLU      │
     │  Dropout p  │
     ├─────────────┤
     │ Linear H→H  │   (×2 hidden blocks by default)
     │   SiLU      │
     │  Dropout p  │
     ├─────────────┤
     │ Linear H→2  │   → [d_x_corr, d_yaw_corr]
     └─────────────┘

  Separately (nn.Parameter, raw scalars):
        b_raw, s_r_raw, asum_raw
   └─► constrain via softplus(+offset)
           b ∈ [0.2, ∞)    (softplus + offset, init ≈ 0.49)
           s_r ∈ (0, ∞)     (softplus, init ≈ 1.0)
           α_sum ∈ (0, ∞)   (softplus, init ≈ 2.0 = α_L+α_R with each ≈1.0)
```

Hyperparameter default: H=128, hidden_blocks=3 (총 4 Linear layers), dropout=0.1.

## 계수 초기값 근거

| 계수 | 초기값 | 근거 |
|------|--------|------|
| `b` (baseline) | 0.49 m | **Scout 2.0 기준 좌우 휠 중심 간 거리(wheel track) = 0.583 m** ([agilexrobotics/ugv\_gazebo\_sim scout\_v2.xacro L18](https://github.com/agilexrobotics/ugv_gazebo_sim/blob/master/scout/scout_description/urdf/scout_v2.xacro)). **Scout Mini wheel track = 0.456 m** ([scout\_mini.xacro L20](https://github.com/agilexrobotics/ugv_gazebo_sim/blob/master/scout/scout_description/urdf/scout_mini.xacro)). 현재 초기값 **0.49 m 는 Scout 2.0 의 전후 축거(wheelbase = 0.498 m)** 에 해당하는 값으로, 차동 구동 키네마틱에서 `b` 는 좌우 휠 중심 간 거리(track)를 사용해야 함. **사용하는 로봇 변형에 맞게 조정 필요: Scout 2.0 → 0.583, Scout Mini → 0.456.** 학습을 통해 실제값으로 역추정되므로 0.49 는 근사 출발값으로만 사용. |
| `s_r` (wheel radius scale) | 1.0 | 내부 odom이 공칭 반경을 사용한다 가정; deviation을 곱셈 스케일로 역추정. |
| `α_sum` (= α_L+α_R) | 2.0 | 좌우 대칭 prior; 개별 `α_L`, `α_R`은 body-frame (d_x, d_yaw) 관측만으로는 식별 불가 (요 방정식에 평균만 등장). 평균 휠 스케일 `α_sum/2` 로만 해석. |

### 물리적 범위 제약
- `b = 0.2 + softplus(b_raw)`의 clamp min, 너무 작은 값 방지
- `s_r = softplus(s_r_raw)` — 양수
- `α_sum = softplus(asum_raw)` — 양수 (좌우 합)
- prior L2 regularization으로 초기값 근처 유지

## 손실 함수 정의

입력/출력 기호:
- `pred = [p_dx, p_dyaw]` (B,2)
- `target = [gt_dx, gt_dyaw]` (B,2)
- `x_raw = [dx_od, dy_od, dz_od, droll_od, dpitch_od, dyaw_od]` (B,6)
- `S_i ∈ {0,1}`: stationary mask

### 1) Data Loss (Weighted MSE)
`d_x`는 m, `d_yaw`는 rad — 스케일 상이. 각 축을 training target std로 정규화하여 단위 편향 제거.

```
L_data = mean_i [ w_x · (p_dx_i - gt_dx_i)^2  +  w_yaw · (p_dyaw_i - gt_dyaw_i)^2 ]
```
기본 `w_x = 1 / (σ_gt_dx^2 + ε)`, `w_yaw = 1 / (σ_gt_dyaw^2 + ε)` (training set fit, 없으면 1.0/1.0).

### 2) Stationary Loss
정지 상태 샘플에서 모든 출력이 0에 가까워야 함.

```
L_stat = mean_i [ S_i · (p_dx_i^2 + p_dyaw_i^2) ] / (mean(S) + ε)
```
ε로 division by zero 방지, stationary ratio가 낮아도 안정.

### 3) Non-holonomic Residual Loss
Scout은 differential drive non-holonomic 플랫폼: body frame에서 lateral velocity는 이상적으로 0. 따라서 보정된 동작 `(p_dx, p_dyaw)` 가 만드는 **per-step body-frame displacement**가 `p_dx · e_x` 방향이어야 하고 `d_y_body ≈ 0` 이 자연스러움.

보정 출력 자체는 lateral을 예측하지 않지만, **raw odom의 `dy_od` 가 큰 경우(횡슬립/센서 노이즈)에도 `p_dx` 가 과도하게 불어나지 않아야** 하고, 물리 모델(계수 스케일링된) 예측과 일관되어야 한다. 구체 residual:

```
# Physics-scaled prediction (단순 1차 근사)
phys_dx    = s_r · dx_od
phys_dyaw  = s_r · dyaw_od · (α_sum / 2) · (b_0 / b)     # b_0 = 초기 baseline 0.49,  α_sum = α_L+α_R

r_x    = p_dx   - phys_dx
r_yaw  = p_dyaw - phys_dyaw

L_nh = mean_i [ r_x^2 + r_yaw^2 ]
```
- `r_x, r_yaw`는 MLP 출력이 물리 모델 예측과 크게 벗어나지 않도록 **soft anchor** 역할. 계수 `(b, s_r, α_sum)` 가 여기서 역추정됨 — residual을 최소화하기 위해 옵티마이저가 계수를 데이터에 맞춰 조정.
- 이전 버전의 lateral penalty `λ_lat · dy_od^2` 는 제거됨. 이 항은 `dy_od` 가 입력 텐서(상수)이므로 **어떤 학습 가능 파라미터에도 gradient 경로가 없어** 실질적 효과가 없었음. Non-holonomic 가정은 대신 모델이 `(d_x, d_yaw)` 만 예측하고 trajectory 누적 시 `d_y = 0` 으로 고정하는 방식으로 자연스럽게 주입됨.

근거: Scout은 body frame에서 수직 방향 속도가 0인 non-holonomic constraint를 가지며, 좌우 휠 속도의 평균 = 선속도, 차이 = 각속도 · baseline 관계를 1차 근사하면 요속(yaw-rate)은 `(v_R − v_L)/b` 형태이나, `(v_L, v_R)` 를 직접 관측할 수 없고 body-frame delta만 있는 본 연구 데이터에서는 **개별 `α_L`, `α_R` 는 식별 불가능** (요 방정식에 `(α_L+α_R)/2` 형태 평균만 등장). 따라서 단일 `α_sum = α_L + α_R` 파라미터로 통합하고 "좌우 평균 스케일 = α_sum/2" 로만 해석한다. 이는 baseline 변화가 각속도 추정에 역비례로 영향을 준다는 정성적 직관을 유지한다.

### 4) Coefficient Prior (L2)
```
L_coeff = (b - 0.49)^2 + (s_r - 1.0)^2 + (α_sum - 2.0)^2
```
데이터 부족 시 계수가 터무니없는 해로 발산하는 것 방지.

### 5) Magnitude Regularization
```
L_mag = mean_i [ p_dx_i^2 + p_dyaw_i^2 ]
```
과도한 보정 방지.

### 최종 손실
```
L = w_d · L_data + w_s · L_stat + w_nh · L_nh + w_c · L_coeff + w_m · L_mag
```

기본 가중치:
| 가중치 | 기본값 | 근거 |
|--------|--------|------|
| `w_d` | 1.0 | 주 task |
| `w_s` | 1.0 | stationary ratio로 정규화된 형태라 안전 |
| `w_nh` | 0.1 | physics anchor — data loss 방해 않도록 작게 |
| `w_c` | 0.01 | prior — 약하게 |
| `w_m` | 0.001 | 마이너 규제 |

## 하이퍼파라미터 탐색 공간 (Optuna/수동)

| 파라미터 | 범위 | 분포 | 기본값 |
|----------|------|------|--------|
| `hidden_dim` | {64, 96, 128} | categorical | 128 |
| `hidden_blocks` | {2, 3, 4} | categorical | 3 |
| `dropout` | [0.0, 0.3] | uniform | 0.1 |
| `learning_rate` | [1e-4, 5e-3] | log-uniform | 1e-3 |
| `coeff_lr_mult` | [0.1, 1.0] | log-uniform | 0.3 (계수 학습률은 MLP보다 낮게) |
| `batch_size` | {128, 256, 512} | categorical | 256 |
| `weight_decay` | [0, 1e-3] | uniform | 1e-5 |
| `w_nh` | [0.01, 1.0] | log-uniform | 0.1 |
| `w_c` | [1e-3, 0.1] | log-uniform | 0.01 |
| `w_m` | [1e-4, 1e-2] | log-uniform | 1e-3 |

## 정규화 전략
| 기법 | 적용 | 파라미터 |
|------|------|----------|
| Dropout | hidden blocks 각각 | p=0.1 |
| Weight Decay | AdamW optimizer | 1e-5 (MLP only) |
| Coefficient Prior | loss term | w_c=0.01 |
| Magnitude Reg | loss term | w_m=1e-3 |
| Early Stopping | training | patience=20 (training-manager가 구현) |
| Gradient Clipping | training | max_norm=1.0 |

## 파라미터 수 추정
기본 H=128, 3 hidden blocks, in=6, out=2:
- Linear(6,128)=896, Linear(128,128)×3=3·16512=49536, Linear(128,2)=258
- 총 ≈ 50,690 + 3 (coeffs) ≈ **50.7k**
- 샘플 수 ≥ 1k 충분 (p/n 비율 관리 양호)

## 베이스라인
1. **Identity**: `pred = [dx_od, dyaw_od]` — 보정 없음. 기본 오차 기준선.
2. **Linear Calibration**: `pred = [a1·dx_od + b1, a2·dyaw_od + b2]` — sklearn `LinearRegression` (입력 2축만).
- `model_module.py`에 `IdentityBaseline`, `LinearBaseline` 클래스로 제공.

## 실험 가설
- H1: Inverse-PINN은 Identity/Linear baseline 대비 ATE, FPE를 낮춘다.
- H2: 학습된 `s_r` 이 1.0에서 유의한 편차 — 내부 odom의 scale bias를 역추정.
- H3: 학습된 `b` 는 0.49 근처에서 소폭 편차 — 플랫폼 고유치 확인.
- H3b: 학습된 `α_sum/2` (평균 휠 스케일) 이 1.0 근처에서 유의 편차 — 좌우 평균 스케일 편이 탐지. (개별 α_L vs α_R 비교는 불가.)
- H4: Physics loss 제거 ablation 시 trajectory-level 메트릭 악화 — physics의 기여 확인.

## 학습관리자 전달 사항
- 옵티마이저: `AdamW`, MLP params와 coeff params에 서로 다른 lr 그룹 (coeff lr은 MLP lr × 0.3).
- Scheduler: `CosineAnnealingLR` 또는 `ReduceLROnPlateau(patience=10)` 권장.
- Grad clip: `max_norm=1.0`.
- Loss 모니터링: `L_data, L_stat, L_nh, L_coeff, L_mag`를 각각 로깅.
- Best checkpoint 기준: `val_L_data` (물리항 제외한 순수 예측 성능).
- 재현성: seed=42, `torch.manual_seed`, `np.random.seed`.
- Coefficient tracking: 매 epoch마다 `b, s_r, α_sum` 값 저장 (해석 가능성 분석용). α_sum/2 를 "평균 휠 스케일" 로 리포트.

## 평가분석가 전달 사항
- 모델 출력은 원단위이므로 그대로 `accumulate_trajectory`에 주입 가능.
- 예상 강점: stationary 구간 안정화, scale bias 보정.
- 예상 약점: 곡선 구간의 고주파 변동, 좌우 비대칭이 시변인 경우.
- 권장 메트릭: MSE(per-axis), ATE, RPE (1s/5s), FPE, Non-holonomic residual std, 학습된 계수 `(b, s_r, α_sum)` 최종 값 + 궤적. `α_sum/2` 를 평균 휠 스케일로 해석.
- Ablation: `w_nh=0`, `w_c=0`, `w_s=0` 각각 비교.

## 리뷰어에게
- 설계 파일: `_workspace/02_model_design.md`
- 구현 파일: `_workspace/experiment_code/model_module.py`
- 데이터 인터페이스: data_module의 `make_tensors` dict 규약 준수
- 계수 초기값 근거, 손실항 가중치 기본값, 활성화 선택 근거를 본 문서에 기재
