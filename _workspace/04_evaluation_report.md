# 04. 평가 보고서 — Inverse-PINN Odometry Correction

## 1. 평가 프레임워크

본 연구의 목표는 Odometry 보정 → 누적 드리프트 감소이므로 평가는 **per-step 오차 + trajectory-level 오차 + 물리적 해석가능성**의 세 축으로 이루어진다. 단일 메트릭(예: MSE)이 성능을 대변하지 못하는 이유는, 회귀 오차가 작더라도 체계적 bias가 있으면 누적 위치 드리프트가 발산할 수 있기 때문이다.

### 1.1 Per-step 메트릭
| 메트릭 | 수식 | 대상 | 의미 |
|--------|------|------|------|
| MAE (d_x) | mean &#124; p_dx − gt_dx &#124; | 회귀 | 평균 거리 오차 (m) |
| RMSE (d_x) | sqrt(mean((p_dx − gt_dx)²)) | 회귀 | 큰 오차에 민감한 척도 |
| MAE (d_yaw) | mean &#124; p_dyaw − gt_dyaw &#124; | 회귀 | 평균 각도 오차 (rad) |
| RMSE (d_yaw) | sqrt(mean((p_dyaw − gt_dyaw)²)) | 회귀 | 큰 오차에 민감한 척도 |
| Identity baseline | odom vs GT | 기준선 | 보정 전 원시 오차 |
| Improvement(%) | (odom − pinn)/odom | 상대 | 모델의 순수 개선도 |

### 1.2 Trajectory 메트릭 (로봇공학 표준)
| 메트릭 | 정의 | 왜 중요한가 |
|--------|------|-----------|
| **ATE_RMSE** (Absolute Trajectory Error) | sqrt(mean(‖pᵢ − pᵢ^gt‖²)) 전구간 | SLAM/odometry 벤치마크의 사실상 표준 (TUM RGB-D, KITTI). 누적 드리프트의 전체 평균. |
| **FPE** (Final Position Error) | ‖p_N − p_N^gt‖ | dead-reckoning 시나리오의 실전 지표 — "얼마나 헤맸나". FPE/path_length(%)로 정규화. |
| **Heading Error RMSE** | RMSE of wrap(yaw − yaw_gt) | 각도 누적 오차; 장기주행에서 위치오차를 지배하는 요인. |
| **RPE_k** (Relative Pose Error) | k-step 상대 변위 오차의 RMSE (k=10) | 국소 일관성 — ATE가 한 번의 큰 오차로 악화돼도 RPE는 국소 품질을 보존해 드러냄. |

채택 근거: ATE와 RPE는 Sturm et al. (2012, IROS) 이래 odometry/SLAM 평가의 표준이며, Scout과 같은 ground robot의 dead-reckoning 연구에서도 직접 적용 가능하다. FPE는 GPS-denied 시나리오의 실용 지표로 함께 보고한다.

### 1.3 물리적 정합성 / 해석가능성
- **Stationary residual**: 정지 샘플에서 예측된 (d_x, d_yaw)의 절대값 평균·최대. 0에 근접해야 함.
- **Coefficient 최종값 및 수렴성**: 마지막 50 epoch 기준 std, coefficient of variation (CV). CV < 5e-3 수렴으로 간주.
- **Scout 공칭값 편차**: (b, s_r, α_sum) − (0.49, 1.0, 2.0). α_sum/2 는 평균 휠 스케일로 해석 (좌우 개별 `α_L, α_R` 는 body-frame (d_x, d_yaw) 관측만으로는 식별 불가).
  - `b` 공칭 참고값: **Scout 2.0 wheel track = 0.583 m** / **Scout Mini wheel track = 0.456 m** ([agilexrobotics/ugv\_gazebo\_sim](https://github.com/agilexrobotics/ugv_gazebo_sim/blob/master/scout/scout_description/urdf/scout_v2.xacro)). 초기값 0.49 m 는 Scout 2.0 전후 축거(wheelbase 0.498 m) 기준이며, 실험 로봇의 변형(Mini vs 2.0)에 맞춰 `B0_INIT` 상수를 조정할 것.

## 2. 기대 결과 (가설)

| ID | 가설 | 예상 지표 | 판정 기준 |
|----|------|----------|---------|
| H1 | PINN이 per-step RMSE(d_x, d_yaw)에서 Identity 대비 개선 | improvement > 0% | 양의 개선 |
| H2 | PINN이 ATE를 Odom-only 대비 감소 | ATE_pinn < ATE_odom | 최소 20% 이상 감소 기대 |
| H3 | FPE가 감소 | FPE_pinn < FPE_odom | 동일 |
| H4 | 학습된 s_r ≠ 1.0 (내부 odom 스케일 bias 역추정) | &#124;s_r − 1&#124; > 0.01 | 유의한 편차 |
| H4b | 학습된 α_sum/2 (평균 휠 스케일) ≠ 1.0 | &#124;α_sum/2 − 1&#124; > 0.01 | 좌우 평균 스케일 편이 |
| H5 | 학습된 b가 0.49 근처에서 안정 | &#124;b − 0.49&#124; < 0.1, last50_std < 5e-3 | 수렴성 |
| H6 | Stationary residual이 0에 근접 | max_abs_dx < 5e-3 m, max_abs_dyaw < 5e-3 rad | 물리 제약 충족 |
| H7 | w_nh=0 ablation 시 trajectory 메트릭 악화 | ATE ↑, FPE ↑ | physics 기여 확인 |

## 3. 성능 요약 (테스트 세트, 실측값 채움 대기)

테이블은 Colab 노트북 실행 후 `metrics.json`에서 자동 채움. 본 보고서의 수치 슬롯은 실행 결과로 치환.

| 모델 | RMSE d_x (m) | RMSE d_yaw (rad) | ATE (m) | FPE (m) | Heading RMSE (rad) | RPE10_trans (m) | Coefficients (b, s_r, α_sum) |
|------|-------------|------------------|---------|---------|-------------------|-----------------|-------------------------------|
| Identity (odom) | TBD | TBD | TBD | TBD | TBD | TBD | — |
| LinearBaseline | TBD | TBD | TBD | TBD | TBD | TBD | — |
| Inverse-PINN | TBD | TBD | TBD | TBD | TBD | TBD | (TBD, TBD, TBD) |

## 4. Ablation 설계

모델 설계에서 도출된 3개 물리 항의 기여도를 측정한다.

| 실험 | 변경 | 기대 효과 |
|------|------|---------|
| Full PINN | 모든 항 활성 | reference |
| `w_nh = 0` | Non-holonomic anchor 제거 | 계수 역추정 signal 상실 → s_r, b가 prior 근처에 머묾; ATE는 data loss만으로 결정 |
| `w_s = 0` | Stationary 항 제거 | 정지 샘플에서 output drift, stationary_residual 상승 |
| `w_c = 0` | Coeff prior 제거 | 데이터 부족 시 계수 발산 — b나 s_r이 물리적 범위 이탈 가능 |

각 ablation은 동일 seed(42), 동일 epochs, 동일 loader로 재학습 후 테스트 trajectory 메트릭을 full 모델과 비교. 노트북 Cell 11에 구현(기본 disabled; `ABLATE=True`로 활성화).

## 5. 노이즈 강건성 (선택, 코드 훅 준비)

`test_loader`의 `x_raw`에 Gaussian noise (σ = 0.1 × std(train odom))를 주입한 복사본 로더를 만들어 `run_full_evaluation`을 재호출한다. ATE/FPE가 noise 부재 대비 얼마나 증가하는지 비교 → 물리항이 없으면 더 크게 악화될 것이라는 가설.

구현 스니펫 예:
```python
# noise injection (skip in headless)
class NoisyDS(torch.utils.data.Dataset):
    def __init__(self, base, sigma):
        self.base = base; self.sigma = sigma
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        item = dict(self.base[i])
        item['x_raw'] = item['x_raw'] + torch.randn_like(item['x_raw']) * self.sigma
        return item
```

## 6. 해석가능성 (XAI)

### 6.1 학습된 계수 vs Scout 공칭값
| 계수 | 공칭값 | 학습값 (TBD) | 편차 | 물리적 해석 |
|------|--------|------------|------|-----------|
| `b` (baseline, m) | 0.49 (근사 출발값) | TBD | TBD | 실제 wheel track(좌우 휠 중심 간 거리)이 공칭치와 얼마나 다른가; 타이어 공기압/차체 조립 오차 반영. **공칭 참고: Scout 2.0 = 0.583 m, Scout Mini = 0.456 m** ([scout\_v2.xacro](https://github.com/agilexrobotics/ugv_gazebo_sim/blob/master/scout/scout_description/urdf/scout_v2.xacro) / [scout\_mini.xacro](https://github.com/agilexrobotics/ugv_gazebo_sim/blob/master/scout/scout_description/urdf/scout_mini.xacro)). 초기값 0.49 는 Scout 2.0 wheelbase(전후 축거) 0.498 m 기반; 사용 로봇에 맞게 조정 필요. |
| `s_r` (wheel radius scale) | 1.0 | TBD | TBD | 내부 odom의 wheel radius 공칭치 대비 실제 유효 반경(마모·압력) |
| `α_sum` (= α_L+α_R) | 2.0 | TBD | TBD | 좌우 합산 휠 스케일. `α_sum/2` 가 평균 휠 스케일 — body-frame (d_x, d_yaw) 관측만으로는 요 방정식에 `(α_L+α_R)/2` 형태 평균만 등장하므로 개별 `α_L`, `α_R` 은 **식별 불가 (non-identifiable)**. 따라서 좌우 비대칭 (`α_R − α_L`) 은 본 데이터/모델에서 추정하지 않으며, 해당 관심사는 좌우 휠 인코더 원신호가 제공될 때의 후속 확장으로 남김 (§9, §10 참조). |

### 6.2 피처 중요도 (회귀 기반 — SHAP 대체)
6차원 입력(d_x, d_y, d_z, d_rolling, d_pitch, d_yaw) 중 MLP의 **Jacobian 기반 기여도**를 후분석으로 계산할 수 있다 (작은 파라미터 수 → 전체 테스트셋에서 해석 가능).
- 예상 상위 피처: `d_x_od → p_dx` (주축), `d_yaw_od → p_dyaw` (주축), `d_y_od → p_dx` (횡슬립 보정, 작음)
- 해석: PINN이 주 입력 채널에 강한 선형-유사 반응을 보인다면 물리 모델과 정합; 비주축 채널에 과도한 의존은 overfitting 신호.

SHAP/LIME은 소형 MLP에서 필수는 아니지만, 대체재로 `captum.attr.IntegratedGradients` 훅을 향후 추가 가능 — 현재 리포트에는 Jacobian-norm ranking 만 정성 권고.

## 7. 배포 준비도

| 항목 | 값 | 판정 |
|------|-----|------|
| 모델 파라미터 수 | ~50.7k (MLP) + 4 (coeffs) | 초경량 |
| 모델 크기 (fp32) | ~0.2 MB | 초경량 |
| 추론 시간 (CPU, batch=1) | <1 ms (측정값 TBD) | 실시간 OK |
| 추론 시간 (CPU, batch=64) | <2 ms (측정값 TBD) | 실시간 OK |
| 메모리 사용 | <50 MB | 엣지 디바이스 OK |
| 배치 처리량 (CPU) | >30k samples/s 예상 | 고빈도 IMU 대응 |
| 결정론성 | seed 고정 + cudnn.deterministic | 재현 가능 |

**배포 관점 권고**: 로봇 온보드 컴퓨팅(예: Jetson Nano, Raspberry Pi 4)에서도 실시간 50 Hz odometry 보정 가능. ONNX 변환 시 `torch.onnx.export(model.mlp, ...)` 형태로 MLP만 내보내고 계수는 상수로 주입.

## 8. 통계적 검증

- **단일 seed 학습이므로 우연성 배제 필요**. 권고: seed ∈ {42, 1337, 2026, 7} 4회 반복 학습 후:
  - ATE/FPE mean ± std 보고
  - Odom vs PINN Wilcoxon signed-rank test (샘플별 per-step error pairwise)
  - p-value < 0.05 + Cohen's d > 0.5 시 유의
- 현재 실행은 단일 seed 결과이며, 실질 통계 검증은 후속 확장으로 배치.

## 9. 한계 및 리스크

| 카테고리 | 한계 | 완화 방안 |
|---------|-----|---------|
| **데이터 범위** | 외부 에러(지면 마찰, 기상, 장애물) 비포함 — 본 연구가 의도적으로 제외 | 외부 에러는 EKF/센서퓨전 상위 계층에 위임 |
| **데이터 규모** | 업로드된 CSV 1세트에 국한 (1k–100k 샘플 예상); 다양한 주행 패턴 검증 부족 | 다양한 시나리오(직선, 곡선, 시작/정지 위주) 데이터 수집 권고 |
| **GT 품질** | 합성 GT(motion capture 시뮬) vs 실측 GT 구분 불명 — 합성이면 내부 bias 패턴 단순화됨 | 실측 GT(Vicon, RTK-GPS) 데이터셋으로 검증 필요 |
| **Non-holonomic 근사** | 실제 Scout은 슬립 발생; trajectory 누적 시 body-frame d_y = 0 고정은 이상화 | slip 추정 보조 피처 도입 혹은 d_y_corr 헤드 추가 고려 |
| **Physics 모델 간이화 & 식별성** | `phys_dyaw = s_r·dyaw·(α_sum/2)·(b0/b)` 는 (v_L, v_R) 직접 관측 부재의 타협. 요 방정식에 `(α_L+α_R)/2` 평균만 등장하므로 좌우 개별 비대칭 `(α_R − α_L)` 은 식별 불가. | 좌우 휠 인코더 직접 관측 가능할 시 `d_x` 식에 `(α_R − α_L)` 이 차동으로 독립 등장하도록 wheel-level 모델로 업그레이드. |
| **계수 time-varying** | b, s_r, α_sum은 시변(마모, 하중)일 수 있으나 현재는 상수로 추정 | 온라인 학습(rolling window EKF hybrid) 확장 |
| **극단적 stationary ratio** | ratio < 1% 시 `L_stat` 신호 약화 | 데이터 수집 시 정지 구간 의도 포함 권고 (data_preparation 문서 반영) |
| **단일 Run** | 통계적 유의성 미검증 | §8 권고 참조 |

## 10. 개선 권고

1. **단기 (현 파이프라인 내)**
   - 다중 seed 평균 리포팅
   - test split trajectory 시작점에서 원점 재설정 후 ATE 계산 (현재 구현됨)
   - `run_full_evaluation` 을 train/val/test 3개 split 모두에 호출해 일반화 간극 관찰
2. **데이터 개선**
   - 좌우 휠 엔코더 원신호 확보 → wheel-level physics 모델로 확장
   - 정지/회전/직선 밸런스된 다중 trajectory 수집
3. **모델 개선**
   - Time-varying 계수(LSTM/GRU hidden state로 s_r(t), b(t) 예측)
   - Uncertainty quantification (MC dropout 또는 heteroscedastic output)
   - Residual 구조화: `pred = odom_scaled + MLP(residual)` 로 prior-centered 예측

## 11. 평가 코드

- `_workspace/experiment_code/eval_module.py` — 구현
- `_workspace/experiment_code/colab_notebook.ipynb` Cell 9~11 — 실행 엔트리포인트
- 출력: `{SAVE_DIR}/test_metrics.json`, `{SAVE_DIR}/test_trajectory.png`, `{SAVE_DIR}/test_error_over_time.png`

Smoke test 완료 (Windows/CPU, torch+matplotlib): random tensor 200 샘플에서 predict→metrics→ATE/FPE→plot→json 까지 에러 없이 완주. (결과 수치는 랜덤 초기화 모델이므로 의미 없음; 파이프라인 검증 용도.)

## 12. 리뷰어 체크리스트

- [x] ATE/RPE/FPE는 로봇공학 표준 메트릭이며 data_module의 `accumulate_trajectory`를 재사용
- [x] Identity baseline 대비 개선도를 모든 메트릭에 병기
- [x] Stationary residual로 물리제약 준수 검증
- [x] 계수 수렴성(last50 std/CV)과 Scout 공칭치 편차를 함께 보고
- [x] Ablation 설계 명시 (w_nh, w_s, w_c)
- [x] 한계(합성 GT, 외부에러 제외, 단일 seed) 명시적 기재
- [x] 배포 관점(경량 모델) 수치화
- [ ] 실데이터 결과값 삽입 (노트북 실행 후 §3, §6.1 수치 채움)
- [ ] 다중 seed 통계 검증 (후속 작업)
