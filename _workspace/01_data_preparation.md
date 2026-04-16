# 01. 데이터 준비 계획 및 파이프라인 (Inverse-PINN Odometry Correction)

## 데이터셋 개요
| 항목 | 값 |
|------|-----|
| 데이터 소스 | 사용자 업로드 CSV 2종 (Colab `files.upload()`) |
| 파일 1 | Ground Truth trajectory (delta 형식) |
| 파일 2 | Robot internal Odometry (delta 형식) |
| 컬럼 | `t, d_x, d_y, d_z, d_rolling, d_pitch, d_yaw` |
| 타깃 변수 | `d_x_corr, d_yaw_corr` (GT의 d_x, d_yaw 로 감독) |
| 입력 변수 | Odom의 `d_x, d_y, d_z, d_rolling, d_pitch, d_yaw` (+ 시간 윈도우) |
| 문제 유형 | 회귀 + 물리 제약 역문제 (Inverse PINN) |
| 총 샘플 수 | 업로드 시점에 결정 (예상 1k–100k 타임스텝) |

## EDA 가이드 (업로드 후 자동 수행)
### 기초 통계 체크리스트
- 각 컬럼: count, mean, std, min, max, % zero, % NaN
- timestamp: 단조 증가 여부, 샘플링 주기(Δt) 평균·분산, 큰 gap(> 3·median Δt) 검출
- 두 파일의 t 범위 교집합 길이, 샘플링 레이트 불일치 여부

### 시각화 (선택)
- d_x, d_yaw 시간 시계열 (GT vs Odom 오버레이)
- delta 히스토그램 → 0 근방 밀도 확인 (정지 구간 비율 추정)
- odom − gt 잔차 분포 → bias 여부 확인

### 다중공선성 / 상관
- d_x ↔ d_yaw 상관은 비선형 커브 시 존재 (직선 구간은 낮음)
- d_y 는 non-holonomic 제약상 0 근처 → variance 낮으면 입력 보조 피처로만 사용

## Timestamp 매칭 전략 (핵심 결정)

### 옵션 비교
| 방법 | 장점 | 단점 | 채택 여부 |
|------|------|------|-----------|
| Nearest-neighbor | 구현 단순, 원본 값 보존 | Δt 차이 크면 aliasing | 1차 채택 |
| Linear Interpolation (delta 재샘플) | 샘플링 레이트 달라도 정밀 매칭 | delta 의 linear interp 는 물리적 의미 제한적 (누적 후 보간이 더 정확) | 옵션 |
| Cumulative → interp → diff | 물리적으로 정합적 | 구현 복잡, 누적오차 반영 | 보조 옵션 |

**채택 근거**: Scout 내부 odom 과 GT (e.g. Motion Capture) 의 샘플링 레이트가 보통 20–100 Hz 범위에서 비슷하고, delta 값은 이미 Δt 로 나눠진 증가분이므로 nearest-neighbor 매칭이 가장 깔끔하다. Δt 차이가 작은 경우 (|Δt_gt − Δt_odom| / Δt_gt < 0.2) nearest, 크면 cumulative-interp-diff 방식으로 폴백한다. 구현은 `load_and_align()` 에서 자동 선택.

### 매칭 허용오차
- `tolerance = 0.5 * median(Δt_odom)` 내 매칭만 유효
- 허용치 밖 샘플은 drop (보고된 drop 비율 기록)

## 정지 검출 (Stationary Mask)

### 기준
- 입력 odometry 의 translational delta 의 L2 norm: `‖(d_x, d_y, d_z)‖ < ε_trans`
- angular delta 의 L2 norm: `‖(d_rolling, d_pitch, d_yaw)‖ < ε_ang`
- 두 조건 모두 만족 시 stationary=True

### 임계값 (기본값; 튜닝 가능)
- `ε_trans = 1e-4` (m per tick; Scout 속도/레이트 고려한 보수값)
- `ε_ang = 1e-4` (rad per tick)
- 실제 값은 EDA 시 노이즈 floor 관찰 후 조정 권장 (`thresh_trans`, `thresh_ang` 파라미터)

### 예상 stationary ratio
- 주행 위주 데이터: 5–15%
- 시작/정지/턴 중심 데이터: 20–40%
- stationary ratio < 1% 이면 physics loss 의 정지 항이 제 역할 못함 → 데이터 수집 시 정지 구간 의도적 포함 권장

## 전처리 파이프라인
| 단계 | 대상 | 변환 | 파라미터 |
|------|------|------|---------|
| 1 | 두 CSV | timestamp 매칭 | nearest / tol=0.5·Δt |
| 2 | 매칭 DF | NaN/Inf drop | - |
| 3 | 입력 odom 6차원 | StandardScaler | **train split 에서만 fit** |
| 4 | 타깃 (d_x, d_yaw of GT) | 스케일 **미적용** | raw 유지 (물리 단위 보존; PINN loss 일관성) |
| 5 | stationary mask 생성 | 임계값 기반 | 위 기준 |

데이터 누수 방지: scaler 는 train split 으로만 `fit`, val/test 는 `transform` 만. 역변환 유틸 제공.

## 데이터 분할 전략
- **시계열**: shuffle 절대 금지 — 시간 순서대로 70% / 15% / 15%
- 연속 블록 분할: `[0 : 0.7N] → train`, `[0.7N : 0.85N] → val`, `[0.85N : N] → test`
- 랜덤 시드: 42 (non-shuffle 이지만 torch generator 용)
- 경계 구간에서의 누적 trajectory 는 각 split 시작점을 원점(0,0,0)으로 재설정하여 평가

선택: 윈도우 기반 샘플링 (window_size=1 기본; 미래 확장용 옵션)

## 피처 엔지니어링 (옵션, 기본 off)
| 신규 피처 | 생성 로직 | 기대 효과 |
|----------|---------|---------|
| speed | sqrt(d_x²+d_y²)/Δt | 비선형 보정 단서 |
| |omega| | |d_yaw|/Δt | 회전 bias |
| is_turning | |d_yaw| > τ | 회전/직선 분기 |

기본 구현에서는 raw 6차원만 사용 (PINN 해석성 우선). 필요 시 모델설계자가 요청.

## 누적 Trajectory 복원
평가 및 시각화용. Scout 은 2D 주행 가정(d_y, d_z, roll, pitch 는 body frame 근사 0).
```
x_{k+1} = x_k + d_x_k · cos(yaw_k) − d_y_k · sin(yaw_k)
y_{k+1} = y_k + d_x_k · sin(yaw_k) + d_y_k · cos(yaw_k)
yaw_{k+1} = wrap(yaw_k + d_yaw_k)
```
`accumulate_trajectory(d_x, d_y, d_yaw)` 유틸 제공.

## 엣지 케이스
- **중복 timestamp**: 먼저 나온 값 유지, 이후 drop
- **역행 timestamp**: drop 후 경고
- **한쪽 파일이 다른쪽보다 짧음**: 교집합 구간만 사용
- **d_yaw 가 −π ~ π 경계 넘나듦**: delta 단위이므로 개별 샘플은 문제없음. 누적 시 `atan2(sin, cos)` 으로 wrapping
- **전부 정지 데이터**: 경고 + 학습 불가
- **NaN/Inf**: 해당 행 drop

## 구현 코드
`_workspace/experiment_code/data_module.py` 참조.

주요 API:
- `upload_data_colab() -> (gt_path, odom_path)`
- `load_and_align(gt_path, odom_path, tol=None, method='nearest') -> pd.DataFrame`
- `detect_stationary(df, thresh_trans=1e-4, thresh_ang=1e-4) -> np.ndarray[bool]`
- `split_timeseries(df, ratios=(0.7,0.15,0.15)) -> (train, val, test)`
- `fit_scaler(train_df, input_cols) -> StandardScaler-like`
- `make_tensors(df, scaler, stationary_mask) -> dict[str, torch.Tensor]`
- `class OdomDataset(Dataset)` + `build_dataloaders(...)`
- `accumulate_trajectory(d_x, d_y, d_yaw, x0=0, y0=0, yaw0=0)`

## 모델설계자 전달 사항
- 입력 shape: `(B, 6)` — `[d_x, d_y, d_z, d_rolling, d_pitch, d_yaw]` (standardized)
- 부가 입력: `stationary_mask (B,)` 불리언, physics loss 의 정지 항 gating 에 사용
- 출력 shape: `(B, 2)` — `[d_x_corr, d_yaw_corr]` (원 단위; scaler 역변환 불필요)
- 학습 가능 시스템 계수 파라미터: `b`(baseline), `s_r`(wheel radius scale), `α_L, α_R`(좌우 비대칭) — 스칼라, 초기값 1.0/0.0 권장
- 비선형성 있으나 입력 범위가 표준화되어 있으므로 tanh/ReLU 기반 small MLP 로 충분
- GT 타깃은 표준화되지 않은 원단위; loss 스케일 주의 (d_x 는 m, d_yaw 는 rad → 스케일 상이 → weighted MSE 권장)

## 학습관리자 전달 사항
- 데이터 로더: `build_dataloaders(batch_size=256, num_workers=0)` — Colab 에서 num_workers=0 권장
- 배치 사이즈 권장: 128–512 (데이터 크기 따라). 시계열 연속성 필요 시 `shuffle=False` (train 도)
- 데이터 볼륨: 입력 크기 작아 in-memory tensor 로 상주 가능 (GB 단위 아님)
- 재현성: `torch.manual_seed(42)`, `np.random.seed(42)`
- 체크포인트 단위: epoch 기준, val loss (physics+data 합산) 기준 best 저장
- stationary mask 는 배치에도 포함시켜 physics loss 계산 시 사용

## 평가분석가 전달 사항
- 클래스 분포 N/A (회귀)
- stationary ratio 보고 필수 (physics loss 의 정지 항 유효성 판단)
- d_y 는 non-holonomic 제약상 거의 0 → 이를 검증 메트릭(non-holonomic residual)으로 활용
- 평가 시 각 split 의 start 에서 누적 복원 → ATE/RPE/FPE 계산
- 노이즈 수준: odom − gt 잔차의 std 로 정량화하여 보고

## 리뷰어에게
- 데이터 누수 없음: scaler fit 은 train only
- 시계열 순서 보존
- physics loss 에 필수인 stationary mask 공급
- 전 파이프라인 재현가능 (seed 고정, 결정적 매칭)
