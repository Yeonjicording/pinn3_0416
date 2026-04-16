# 05. 실험 리뷰 보고서 — Inverse-PINN for Scout Odometry Correction

**리뷰 일자**: 2026-04-15
**리뷰어**: experiment-reviewer
**대상**: `_workspace/` 전 산출물 + `experiment_code/` 전 모듈

---

## 1. 실험 개요

Agilex Scout (differential drive, non-holonomic) 플랫폼의 내부 odometry 오차를 보정하기 위한 **Inverse Physics-Informed Neural Network (Inverse-PINN)** 연구. MLP가 `(d_x_corr, d_yaw_corr)`를 직접 예측하면서, 학습 가능한 시스템 계수 `(b, s_r, α_L, α_R)`를 physics loss의 gradient를 통해 역추정한다.

- **문제 유형**: 회귀 + 역문제 (시스템 계수 역추정)
- **입력**: Scout body-frame delta 6D `(d_x, d_y, d_z, d_rolling, d_pitch, d_yaw)` odom
- **출력**: 보정된 `(d_x_corr, d_yaw_corr)` (원단위, m/rad)
- **감독**: Ground truth delta `(gt_d_x, gt_d_yaw)`
- **물리 제약**: Stationary 항, Non-holonomic anchor 항, Coefficient prior, Magnitude reg
- **제외**: 외부 에러(지면/기후/장애물) 의도적 배제

---

## 2. 산출물 인벤토리

| 경로 | 내용 | 상태 |
|------|------|------|
| `_workspace/00_input.md` | 사용자 입력 요구사항 정리 | OK |
| `_workspace/01_data_preparation.md` | 데이터 파이프라인 설계 | OK |
| `_workspace/02_model_design.md` | 아키텍처·손실·하이퍼파라미터 | OK |
| `_workspace/03_training_config.md` | 학습 설정·재현성·체크포인트 | OK |
| `_workspace/04_evaluation_report.md` | 평가 프레임워크 + TBD 수치 슬롯 | OK(수치 미채움) |
| `experiment_code/data_module.py` | 정렬·분할·스케일러·`OdomDataset`·`accumulate_trajectory` | OK + selftest 통과 |
| `experiment_code/model_module.py` | `InversePINN`, `PhysicsLoss`, 베이스라인 2종 | OK + smoke 통과 |
| `experiment_code/train_module.py` | `TrainConfig`, `train_api`, `load_best`, 로거, 조기종료 | OK + smoke 통과 |
| `experiment_code/eval_module.py` | per-step/ATE/RPE/FPE/stationary/coeff/플롯 | OK + smoke 통과 |
| `experiment_code/colab_notebook.ipynb` | 12 셀 end-to-end | OK |

---

## 3. 정합성 검증 결과

### 3.1 배치 키 스키마 (data → model → loss → train → eval)

| 키 | data_module | model_module | train_module | eval_module | 일치 |
|---|---|---|---|---|---|
| `x` (B,6 std) | `make_tensors`/`OdomDataset.__getitem__` | `InversePINN.forward(x)` | `_model_forward(batch["x"])` | `predict_deltas` uses `batch["x"]` | ✓ |
| `x_raw` (B,6 raw) | 제공 | `PhysicsLoss(..., x_raw=...)`, `IdentityBaseline.forward(x, x_raw=)` | 전달 | `batch["x_raw"]` 수집 | ✓ |
| `y` (B,2) | `[gt_d_x, gt_d_yaw]` 원단위 | target | `batch["y"]` | `gts[:,0], gts[:,1]` → `gt_dx, gt_dyaw` | ✓ |
| `stationary` (B,) bool | `detect_stationary(side='od')` | `_loss_stationary`, `_loss_nonholonomic` nonstat gate | 전달 | `stats` 수집 | ✓ |
| `dt` (B,) | 제공 (첫 원소 보정) | 미사용 | 미사용 | 미사용 | ✓ (사용처 없음; 미래 확장용) |

**결과**: 키 이름/형상 완전 일치. 정합성 이상 없음.

### 3.2 `x` vs `x_raw` 사용 규칙

- MLP 입력: **standardized** `x` — 정석 (`model_module.InversePINN.forward` / `_MLP.net`)
- Physics loss non-holonomic anchor: **x_raw** — `dx_od, dy_od, dyaw_od` 사용 ✓
- Identity/Linear baseline: **x_raw** 사용 (원단위 예측) ✓
- Target `y`: 원단위 (스케일러 미적용) — MLP 출력이 원단위로 나와야 한다는 설계와 합치

**결과**: 원단위/표준화 구분 규칙 일관적으로 준수됨.

### 3.3 `accumulate_trajectory` API 공유

- `data_module.accumulate_trajectory(d_x, d_y, d_yaw, x0, y0, yaw0)` — 정식 구현
- `eval_module._get_accumulate()` — `data_module` 우선 import, 실패 시 로컬 `_accumulate` fallback
- 로컬 fallback의 signature 및 Euler 적분·yaw wrapping 로직이 data_module 구현과 바이트 수준으로 동일 (2D Scout 가정)

**결과**: API 동일. 중복이지만 의존성 실패 방어책으로 합리.

### 3.4 모델-데이터 연결

- `INPUT_COLS = ["d_x","d_y","d_z","d_rolling","d_pitch","d_yaw"]` (data_module)
- `IDX_DX=0, IDX_DY=1, IDX_DYAW=5` (model_module) — 일치
- eval_module `predict_deltas`에서 `odom_dyaw = odoms[:, 5]` — 일치

**결과**: 인덱스 매핑 완전 일치.

### 3.5 학습-평가 계수 전달

- `train_module.train` → `coefficients`가 `best.pt`에 저장됨 (해당 epoch 시점의 값)
- `eval_module.run_full_evaluation(final_coeffs=ckpt['coefficients'])` 전달 ✓
- 노트북 Cell 9가 `ckpt.get('coefficients')` 전달 ✓

**주의**: `best.pt`의 `coefficients`는 **best epoch 시점** 값. `train_api`가 반환하는 `final_coefficients`는 **학습 종료 시점** 값. 두 값이 다를 수 있음. 현재 노트북·eval_module은 `ckpt['coefficients']`(=best) 사용 — 설계 문서와 정합적이지만 차이 존재를 명시 권장.

---

## 4. 과학적 엄밀성 평가

### 4.1 Inverse-PINN 방법론 타당성

- MLP는 `[d_x_corr, d_yaw_corr]`를 직접 예측 (Option A). 계수 `(b, s_r, α_L, α_R)`는 nn.Parameter로 등록되어 physics loss를 통해서만 gradient를 받음.
- `L_nh`의 `phys_dx = s_r · dx_od` 항은 `s_r`에 대한 유의미한 gradient를 생성함. Data loss와 non-holonomic anchor의 tension을 통해 `s_r`이 `mean(gt_dx / dx_od)` 방향으로 역추정됨.
- `L_coeff` (prior L2)가 없으면 data 부족 시 발산 가능성 — 본 설계는 `w_c=0.01`로 약한 anchoring 제공 ✓

**결과**: Inverse 구조는 원리상 작동한다.

### 4.2 Physics loss 수식 엄밀성

차동 구동(differential drive)의 정확한 kinematic:
```
v = (v_R + v_L)/2
ω = (v_R − v_L)/b
d_x   = v · dt
d_yaw = ω · dt = (v_R − v_L)·dt / b
```

본 구현의 `phys_dyaw`:
```
phys_dyaw = s_r · dyaw_od · ((α_L + α_R)/2) · (B0_INIT / b)
```

문제점:
- **α_L, α_R 개별 식별 불가능(unidentifiable)**: 식에 `(α_L + α_R)/2` 평균으로만 등장. 좌우 비대칭(v_R − v_L)은 `α_R − α_L`로 표현되어야 하지만 여기서는 평균 스케일로만 작용. 따라서 `α_L, α_R`는 개별적으로는 L_coeff prior로만 제약받아 **초기값 근처에서 실질적으로 이동하지 않을 가능성이 큼**. 문서(02_model_design §Physics loss) 마지막 문단이 이 한계를 이미 일부 언급하지만, 최종 리포트에서 "α_L, α_R 개별 값이 해석가능하다"고 주장하는 것은 **엄밀하지 않음**.
- **b의 부호·스케일 약식**: `B0_INIT/b`는 "b가 커지면 d_yaw 감소" 정성적 직관은 맞으나, 실제는 `dyaw_od`가 이미 내부 b_internal로 나눠진 값이므로 스케일 팩터 `b_internal/b_true`여야 함. 여기서 `B0_INIT=0.49`를 `b_internal`로 가정 — **가정 명시 필요**.
- `r_lat = dy_od`만 penalize하는 방식은 입력 신호(관측된 횡슬립)에만 벌을 주는 것이지 모델 출력과 무관 — 현재 gradient는 오직 `coeffs`(없음)와 `x_raw`(leaf 아님)로만 흐르므로 **실질적으로 학습에 기여하지 않는 상수 항**. 이는 L_nh 전체의 scalar 값에만 영향. 수정 필요 (예: `p_dy` 예측 추가 또는 제거).

### 4.3 Data loss weight 스케일

- `compute_data_weights`: `w = 1/var(gt)`. d_x가 ~0.1 단위(m/tick), d_yaw가 ~0.02(rad/tick)라면 `w_yaw >> w_x`가 되어 균형 맞음 ✓
- `w_data=1.0, w_stationary=1.0, w_nonholonomic=0.1, w_coeff=0.01, w_magnitude=1e-3` — 데이터 셋별 편차 발생 가능하나, 문서에 탐색 범위 명시됨 ✓

### 4.4 Magnitude + Stationary 중복

- `L_stat`: 정지시 `pred^2`에 `w_s=1.0` × `1/mean(S)` ≈ O(10) 가중
- `L_mag`: 전구간 `pred^2`에 `w_m=1e-3`

정지 샘플은 양쪽에서 penalize되지만, L_mag 계수가 매우 작아 실질 중복 영향 미미. 의도는 명확(정지항은 정지 샘플 집중, mag은 전구간 일반).

### 4.5 합성 smoke test의 한계

- `data_module._selftest`: 간단한 circular motion + scale bias(1.1, 0.95)로 시뮬
- `model_module._smoke_test`: 완전 random 입력
- `train_module._smoke_test`: 5 epoch 짧은 루프 — 수렴 보장 X
- `eval_module._smoke_test`: random 모델 → metrics 계산 파이프라인만 검증

**중요**: 이들은 모두 **파이프라인 에러-프리**를 검증할 뿐, **실제 학습 품질을 보장하지 않음**. 실측 데이터 없이는 H1~H7 가설 어느 것도 확인 불가. 04 평가 보고서 §3, §6.1의 "TBD" 슬롯이 전부 비어있는 현 상태를 **명시적 한계**로 간주해야 한다.

---

## 5. 재현성 평가

| 항목 | 상태 | 비고 |
|------|------|------|
| `torch.manual_seed(42)` | ✅ | `set_seed` 공통 유틸 |
| `numpy`, `random`, `PYTHONHASHSEED` 동기화 | ✅ | set_seed에서 일괄 |
| `cuda.manual_seed_all` | ✅ | CUDA 시 |
| `cudnn.deterministic=True`, `benchmark=False` | ✅ | `deterministic=True` 기본 |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | ✅ | setdefault로 설정 |
| 환경 dump (`config.json`) | ✅ | `TrainConfig` asdict + extra |
| Scaler fit은 **train split only** | ✅ | `prepare_all`에서 `fit_scaler(df_tr, ...)` — 데이터 누수 없음 |
| 시계열 분할 (shuffle 없음) | ✅ | `split_timeseries` 연속 블록, `shuffle_train=False` 기본 |
| 체크포인트 재현 (`load_best`) | ✅ | optimizer/model/coeffs/config 전부 저장 |
| 단일 seed 결과만 | ⚠️ | §9 한계에 명시 — 다중 seed 권장 (후속) |
| AMP와 결정론 양립 | ⚠️ | AMP(fp16)는 엄밀한 deterministic 보장 X — deterministic+amp 동시 활성화 시 미세 변동 가능 |
| Colab 노트북 재실행 | ✅ | 상대 경로, 모듈 reload, 업로드 UX 검증 |

---

## 6. 🔴 필수 수정 항목

### 6.1 `PhysicsLoss._loss_nonholonomic`의 `r_lat` 항이 학습에 기여하지 않음
**파일**: `model_module.py:325-328`
```python
r_lat = dy_od * nonstat
...
l_lat = (r_lat ** 2).mean()
return l_anchor + self.cfg.lambda_lat * l_lat
```
`dy_od`는 입력 텐서이므로 gradient를 만들지 않음. `r_lat^2`는 loss 값에만 영향을 주는 **상수항** (`pred`, `coeffs`로의 backprop 경로 없음).

**영향**: `λ_lat`가 0이든 1이든 모델 학습에 영향 없음. 문서(02_model_design §Non-holonomic)에서 "d_y noise가 큰 구간에서 d_x 예측 변동 증가"를 주장하지만 현 구현은 이를 실현하지 못함.

**제안**: 모델에 `d_y_corr` 출력 헤드 추가 후 `r_lat = p_dy_corr * nonstat` 또는, pred의 cross-term으로 연결 (예: `r_lat = (p_dx * sin(p_dyaw) - p_dx * 0) * nonstat` 등 물리 의미 있는 식). 대안으로 **해당 항을 제거하고 문서·탐색공간에서 `λ_lat`를 삭제**하는 것이 가장 정직.

### 6.2 `α_L`, `α_R` 개별 식별 불가능(unidentifiable)
**파일**: `model_module.py:316`
```python
phys_dyaw = s_r * dyaw_od * ((aL + aR) * 0.5) * (B0_INIT / b.clamp_min(1e-3))
```
두 파라미터가 항상 `(aL + aR)/2`로만 등장 → 합(또는 평균)만 식별 가능, **개별값은 prior L2에만 의존**. 04 평가 보고서 §6.1이 `α_L, α_R` 개별 추정치와 `α_R − α_L` 편향 해석을 제공하려 하지만 현 수식상 불가능.

**영향**: 학습 후 `α_L ≈ α_R ≈ 1.0`이 거의 확실 (prior에 의해). 해석가능성 주장 과장.

**제안(택1)**:
- (a) `α_L, α_R`를 `α_sum = (α_L+α_R)/2` 단일 파라미터로 축약하고 04 평가 보고서·02 설계 문서 수정.
- (b) 수식을 실제 differential drive에 맞게 `phys_dx = s_r · (α_R + α_L)/2 · dx_od`, `phys_dyaw = s_r · (α_R − α_L) · (something) / b + s_r · dyaw_od · (b0/b)` 식으로 개별 식별 가능하게 재설계 (단, 관측 `(v_L, v_R)` 없이는 한계).
- (c) 현 수식 유지 시 04 평가 보고서 §6.1에 **"α_L, α_R 개별값은 식별 불가능; 평균만 역추정됨"**을 명시.

### 6.3 Scout 2.0 baseline 0.49m 출처 미인용
**파일**: `02_model_design.md:56`, `model_module.py:56` (`B0_INIT = 0.49`)

"Agilex Scout 2.0 공식 datasheet 기준 wheel-track ≈ 0.498 m"이라고 하나 URL/문서 버전이 인용되지 않음. 재현성·학술 신뢰성 저하.

**제안**: Agilex 공식 product page 또는 manual PDF URL을 `02_model_design.md`에 인용. 값이 확인되지 않으면 추정치임을 명시(`0.49 ± 0.05 m (추정)`).

---

## 7. 🟡 권장 개선

1. **단일 seed**: §8에 명시된 대로 `{42, 1337, 2026, 7}` 다중 seed run 수행 후 ATE/FPE mean±std 보고. 노트북에 루프 추가 (약 4×학습 시간).

2. **Ablation 자동 실행**: Cell 11이 주석 처리됨. 최종 리포트 작성 시 `ABLATE=True`로 실행하여 04 §4의 ablation 표 채우기.

3. **`TrainConfig` 기본 `batch_size` 필드 부재**: 03 training_config는 "Batch Size 64"를 명시하나 `TrainConfig` dataclass에 `batch_size` 필드가 없음. 노트북 Cell 4에서 별도 `BATCH_SIZE=64` 상수로 지정. 일관성 위해 `TrainConfig`에 필드 추가 또는 문서에서 명시하여 혼동 방지.

4. **AMP + deterministic 양립성 경고**: `amp=True, deterministic=True` 동시 사용 시 fp16의 비결정성으로 완전한 bit-reproducibility는 보장 X. 엄밀 재현이 필요한 run에서는 `amp=False` 권장을 03 문서에 명시.

5. **Stationary 검출 기준 튜닝 가이드**: `thresh_trans=1e-4`가 실제 Scout 내부 odom 노이즈 floor와 맞지 않을 수 있음. 노트북 Cell 4에서 delta norm 히스토그램을 출력해 사용자가 임계값을 보정하는 셀 추가 권장.

6. **`dt` 채널 미사용**: data_module이 `dt`를 제공하나 모델/손실에서 사용하지 않음. 비균일 샘플링 데이터의 경우 `phys_dx = s_r · v_od · dt` 형태로 `dt`를 곱해야 물리적으로 올바름. 샘플링이 균일하다고 가정하면 무해하나 가정 명시 필요.

7. **`b_internal` 가정 명시**: `phys_dyaw = ... · (B0_INIT/b)` 수식은 내부 odom이 `b_internal = B0_INIT = 0.49`를 사용한다는 암묵적 가정에 의존. 02 설계 문서에 명시 권장.

8. **노트북 Cell 8과 Cell 9의 지표 중복**: Cell 8에서 수동 MSE/MAE 계산 후 Cell 9에서 `per_step_metrics`가 동일 지표 재계산. Cell 8을 제거하고 Cell 9~10만 유지해도 무관.

9. **Colab Cell 2에서 eval_module.py 동반 업로드**: 현재 `data_module, model_module, train_module` 3개만 업로드하고 Cell 9에서 `eval_module`을 별도 업로드함. UX 개선: Cell 2에서 4개 모두 업로드 지시.

10. **`coefficient_summary`의 `final`과 `convergence.last`**: 전자는 전달된 `final_coeffs` (= best checkpoint 시점), 후자는 history 기반 마지막 epoch — 두 값이 다를 수 있음. 04 평가 보고서의 "학습된 계수"라는 표현이 어느 쪽을 의미하는지 명시 필요.

---

## 8. 🟢 잘된 점

1. **배치 키 스키마 완전 일관**: 5개 모듈 전반에 걸쳐 `x, x_raw, y, stationary, dt` 규약이 깨지지 않음.
2. **데이터 누수 방지**: Scaler는 train split only로 fit, val/test는 transform만. 시계열 분할 순서 보존.
3. **체크포인트 완전성**: state/optimizer/coeffs/config/val_data 전부 저장 → 완전 재현 가능.
4. **평가 프레임워크**: ATE/RPE/FPE는 로봇공학 표준(Sturm et al. 2012) 인용, accumulate_trajectory로 구현 통일.
5. **베이스라인 제공**: Identity + LinearBaseline 2종 — 보정 효과 정량화 기준선 확보.
6. **Loss 분해 로깅**: `train/val` × 5개 loss 항 × coeff 히스토리 개별 기록 → ablation·디버깅 용이.
7. **Early stopping이 `val_data`에만 의존**: physics 가중치 튜닝으로 인한 편향 방지 (설계적으로 탁월).
8. **Softplus+floor로 b 양수화**: 물리적 범위 자연스럽게 강제.
9. **Colab fallback UX**: GPU 없으면 CPU + AMP 자동 비활성. `resolve_device("auto")`.
10. **Smoke test 4종**: 각 모듈에 `_smoke_test()` 내장 → CI/빠른 검증 용이.
11. **AMP 시 `clip_grad_norm`을 `unscale_` 이후 적용**: 올바른 순서 (PyTorch 관행 준수).
12. **data_module `cumdiff` 옵션**: 샘플링 레이트 불일치 시 물리적으로 정합한 alignment 제공 (사용은 선택).

---

## 9. 사용자를 위한 실행 가이드 (Colab)

### 9.1 준비
1. Colab 새 노트북 생성 (메뉴 Runtime → Change runtime type → GPU 선택; CPU도 동작).
2. 로컬 `experiment_code/` 디렉토리에서 4개 파일 준비: `data_module.py`, `model_module.py`, `train_module.py`, `eval_module.py`.
3. 원본 CSV 2개 준비: GT(ground truth) 파일명에 `gt` 포함, Odom 파일명에 `odom` 포함 권장 (자동 분류). 컬럼: `t, d_x, d_y, d_z, d_rolling, d_pitch, d_yaw`.

### 9.2 셀별 실행 순서 (colab_notebook.ipynb)

| 셀 | 동작 | 주의 |
|----|------|------|
| Cell 1 | torch/numpy/pandas/matplotlib 버전 체크 | 모두 Colab에 사전설치; 실패시 자동 설치 |
| Cell 2 | `data_module.py`, `model_module.py`, `train_module.py` 업로드 (Option B) — **또한 `eval_module.py`도 함께 업로드 권장** (Cell 9의 별도 업로드 스킵 가능) |
| Cell 3 | GT + Odom CSV 2개 업로드 | 파일명에 `gt`/`odom` 포함 시 자동 분류 |
| Cell 4 | align/split/scaler/loader/stationary ratio 출력 | **stationary ratio < 1% 경고** 발생 시 threshold 조정 필요 |
| Cell 5 | InversePINN + PhysicsLoss 빌드, smoke forward | `init coeffs`가 (b≈0.49, s_r≈1.0, α_L≈1.0, α_R≈1.0)인지 확인 |
| Cell 6 | **학습 (최대 300 epochs, patience=20)** | GPU T4에서 1k 샘플 수 분 / CPU에서 수십 분 |
| Cell 7 | Loss curve + coefficient trajectory 시각화 | `train_data`/`val_data` 발산이면 Cell 6 재학습 (mlp_lr=5e-4, amp=False) |
| Cell 8 | best 체크포인트 로드 + test 수동 MSE/MAE | Cell 9와 중복 — 생략 가능 |
| Cell 9 | eval_module 업로드 (Cell 2에서 이미 업로드했으면 스킵) | |
| Cell 10 | `run_full_evaluation` → metrics.json + PNG 2개 | ATE/FPE/Heading 지표 출력 |
| Cell 11 | 결과 표 출력 + improvement% | PINN vs Odom 비교 |
| Cell 12 | 궤적·에러 PNG 인라인 표시 | |
| Cell 13~ (옵션) | Ablation (w_nh=0, w_s=0, w_c=0) | `ABLATE=True`로 활성화; 3× 학습 시간 |

### 9.3 실패 시 디버깅 체크리스트
- **NaN/Inf loss**: `TrainConfig(amp=False, mlp_lr=5e-4, grad_clip_max_norm=0.5)` 재시도.
- **stationary ratio 0%**: 정지 구간 없는 데이터 → `L_stat` 효과 없음. `thresh_trans/thresh_ang` 상향 또는 데이터 수집 재검토.
- **OOM**: `BATCH_SIZE=32` 하향.
- **coefficient 발산**: `w_coeff=0.1`로 prior 강화.

### 9.4 결과물 저장 위치
- `/content/runs/pinn/best.pt` — 모델 + 계수 + config
- `/content/runs/pinn/history.json` — epoch/metric/coeff 이력
- `/content/runs/pinn/test_metrics.json` — 평가 지표
- `/content/runs/pinn/test_trajectory.png`, `test_error_over_time.png` — 그래프

Drive 저장: Cell 6 이전 `SAVE_DIR='/content/drive/MyDrive/...'`로 변경 후 `drive.mount` 실행.

---

## 10. 최종 판정

### 정합성 매트릭스
| 검증 항목 | 상태 | 비고 |
|-----------|------|------|
| 데이터 ↔ 모델 (키/형상/인덱스) | ✅ | 완전 일관 |
| 모델 ↔ 학습 (forward/coeffs/optimizer 그룹) | ✅ | 완전 일관 |
| 학습 ↔ 평가 (체크포인트/history/accumulate) | ✅ | best vs final 차이 명시 필요 |
| 재현 가능성 | ✅ | seed/결정론/환경 dump — AMP deterministic은 경고 |
| 데이터 누수 검증 | ✅ | scaler fit train-only, time-order 보존 |
| Physics loss 수학적 엄밀성 | ⚠️ | `r_lat` 상수항 버그, α_L/α_R 개별 식별 불가 |
| 실험 증거 (실측 결과) | ⚠️ | TBD — 사용자 데이터 업로드 필요 |
| 단일 seed | ⚠️ | 권장 후속 |

### 실험 결과 요약
| 모델 | 주요 메트릭 | 베이스라인 대비 | 통계적 유의성 |
|------|-----------|-----------------|----------------|
| Identity | TBD | — | — |
| Linear | TBD | TBD | — |
| Inverse-PINN | TBD | TBD | TBD (단일 seed) |

(실측 데이터 없이 학습이 수행되지 않아 모든 수치 슬롯이 비어 있음. 파이프라인 자체의 smoke test는 모두 통과.)

### 최종 판정: **NEEDS-FIX (경미)**

**근거**:
- 파이프라인·정합성·재현성 관점에서는 **READY 수준**. 네 개 모듈이 서로 키 규약을 완벽히 지키고, smoke test가 에러 없이 통과하며, Colab 노트북이 clean 환경에서 실행 가능한 구조를 갖춤.
- 다만 **필수 수정 3건**(6.1 `r_lat` 상수항, 6.2 `α_L, α_R` 미식별, 6.3 Scout 0.49m 출처)은 논문·리포트 제출 전 반드시 해결해야 하는 **과학적 엄밀성 이슈**.
- 6.2의 경우 **수식 재설계** 또는 **해석 주장 축소** 중 택1만 하면 즉시 해결.
- 6.1은 코드 수정 또는 항 제거로 해결.
- 6.3은 인용 추가 또는 "추정치" 명시로 해결.
- 수치 결과는 사용자 실제 데이터 업로드 후 Colab 실행으로 채워진다. 그 후 다중 seed + ablation까지 수행하면 논문 수준 품질 달성 가능.

### 후속 실험 제안
1. (🔴 기반) 6.1/6.2 수정 후 **α_sum 단일 파라미터화** 또는 **wheel-level 관측** 확장으로 재학습.
2. (🟡 후속) 다중 seed `{42, 1337, 2026, 7}` 4회 run → Wilcoxon signed-rank + Cohen's d 통계 검증.
3. (🟡 후속) Ablation matrix 자동화 (full, no_nh, no_stat, no_coeff) — 04 §4 표 채움.
4. (🟢 확장) Noise injection 실험 (04 §5)으로 physics loss의 강건성 검증.
5. (🟢 확장) Time-varying 계수(LSTM/GRU hidden으로 s_r(t), b(t) 예측)로 확장 — 마모·하중 변화 대응.
6. (🟢 확장) 실측 Scout 데이터셋 수집 시 좌우 휠 엔코더 원신호 `(v_L, v_R)` 확보 → 진짜 개별 `α_L, α_R` 식별 가능.

---

**리뷰어 서명**: experiment-reviewer
**리뷰 모드**: 1회 독립 검증 (재작업 요청 없음)

---

## 11. 재검증 결과 (Round 2)

**재검증 일자**: 2026-04-15
**범위**: Round 1 에서 지적된 🔴 필수 수정 3건의 반영 여부

### 11.1 항목별 판정

| # | 🔴 항목 | 상태 | 증거 |
|---|---------|------|------|
| 6.1 | `PhysicsLoss._loss_nonholonomic` 의 gradient-dead `r_lat` 항 제거 + `LossConfig.lambda_lat` 삭제 | **RESOLVED** | `model_module.py:319-351` `_loss_nonholonomic` 는 `l_anchor = (r_x^2 + r_yaw^2).mean()` 만 반환, `dy_od` 미사용. `LossConfig` dataclass 필드에 `lambda_lat` 부재 (smoke test: `'lambda_lat' in LossConfig.__dataclass_fields__ → False`). 주석(`model_module.py:345-349`)에 제거 사유 명시. `02_model_design.md:106` 도 제거 근거 서술. |
| 6.2 | `α_L, α_R` → 단일 `alpha_sum` 통합 및 전파 | **RESOLVED** | `model_module.py`: `ALPHA_SUM_INIT = 2.0`, `asum_raw` 파라미터, `coefficients() = {b, s_r, alpha_sum}`. `_loss_nonholonomic` 에서 `alpha_sum * 0.5` 사용. `_loss_coeff` 도 `(alpha_sum - alpha_sum_prior)^2` 로 갱신. `eval_module.py:301,316-324`: `keys/priors` 에 `alpha_sum` 반영 + `alpha_mean = alpha_sum/2` 파생 지표 추가. `train_module.py:430`: 로그 `asum=` 출력. `colab_notebook.ipynb:269,274`: 플롯/프린트 키 `('b','s_r','alpha_sum')`. `02_model_design.md:47,58,98,112` 및 `04_evaluation_report.md:30` 모두 `α_sum` 단일 파라미터 표기 + 비식별성 해석 명시. Smoke test: coef keys `['b','s_r','alpha_sum']`, `asum_raw.grad` 비영(gradient 경로 확인). |
| 6.3 | Scout baseline 출처 보강 + 0.49 m 해석 경고 | **RESOLVED** | `model_module.py:61-79`: Scout 2.0 wheel track 0.583 m / Scout Mini 0.456 m 공식 URDF URL(agilexrobotics/ugv_gazebo_sim) 인용, 0.49 m 는 Scout 2.0 wheelbase (0.498 m) 에 해당한다는 경고 포함. `02_model_design.md:56` 동일 내용 반영. `04_evaluation_report.md:30-31` 에 Scout 2.0/Mini track 값 및 URDF URL 인용, 0.49 근거 경고 추가. |

### 11.2 🟡 잔여 정합성 이슈 (PARTIAL, 비차단)

| 위치 | 문제 | 권장 조치 |
|------|------|----------|
| `03_training_config.md:26` | "Loss 가중치" 행에 `lambda_lat=0.1` 잔존 | `lambda_lat` 표기 제거 — 실제 `LossConfig` 에 필드 부재이므로 문서 불일치. |
| `04_evaluation_report.md:131` | 리스크 테이블 "Non-holonomic 근사" 행 완화책에 `λ_lat 튜닝` 기재 | `d_y=0` 고정 방식 또는 `p_dy` 출력 헤드 추가 등으로 문구 교체. |

두 항목 모두 **문서 잔여 참조**로, 실행/학습 결과에 영향 없음. 독자의 혼동 방지 목적에서만 정정 필요.

### 11.3 Python Smoke Test (CPU)

```
smoke OK total= 0.686
parts= {'total': 0.686, 'data': 0.454, 'stationary': 0.000,
        'nonholonomic': 2.324, 'coeff': 0.000, 'magnitude': 9.9e-4}
asum_raw grad nonzero: True
b_raw grad nonzero: True
LossConfig has lambda_lat: False
coef keys: ['b', 's_r', 'alpha_sum']
```

- `nonholonomic` 손실이 유의미한 스칼라 값(2.324) 이며, `asum_raw`·`b_raw` 모두 gradient 수신 → 항이 **실제 역전파에 기여** 함이 검증됨 (이전 `r_lat` 상수항 버그 완전 해소).

### 11.4 정합성 매트릭스 (Round 2)

| 항목 | Round 1 | Round 2 |
|------|---------|---------|
| 데이터 ↔ 모델 | ✅ | ✅ |
| 모델 ↔ 학습 | ✅ | ✅ (alpha_sum 키 일관성 확인) |
| 학습 ↔ 평가 | ✅ | ✅ (`alpha_sum` priors/keys 전파) |
| 재현 가능성 | ✅ | ✅ |
| 데이터 누수 검증 | ✅ | ✅ |
| Physics loss 수학적 엄밀성 | ⚠️ | ✅ (6.1/6.2 해소, gradient 유효) |
| Baseline 값 출처 인용 | ❌ | ✅ (공식 URDF URL 인용) |

### 11.5 최종 판정

**READY**

- Round 1 의 🔴 필수 수정 3건 모두 **RESOLVED**.
- 두 건의 🟡 문서 잔여 참조(03_training_config.md, 04_evaluation_report.md 각 1줄)는 비차단 권장 수정이며, 실제 실행·학습·평가 파이프라인의 정합성에는 영향이 없음.
- Python CPU smoke test 통과, 계수 gradient 흐름 확인, LossConfig 필드 정합 확인.
- 실측 데이터 업로드 및 다중 seed 학습 단계로 진행 가능.

