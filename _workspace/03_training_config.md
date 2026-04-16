# 03. 학습 설정 및 실험 추적 — Inverse-PINN

## 실험 추적 설정
- **플랫폼**: In-process JSON logger (lightweight; Colab 친화적, 외부 계정 불필요). 선택적으로 MLflow/W&B wrapper 교체 가능.
- **프로젝트명**: `inverse_pinn_scout`
- **실험명 규칙**: `{dataset_tag}_{model_tag}_{seed}_{timestamp}` (예: `ds_v1_pinnH128_s42_20260415`)
- **로깅 항목**:
  - 메트릭(epoch 단위): `train_total, train_data, train_stationary, train_nonholonomic, train_coeff, train_magnitude, val_total, val_data, val_stationary, val_nonholonomic, val_coeff, val_magnitude, train_elapsed_s, cuda_mem_alloc_gb, cuda_mem_peak_gb, lr_mlp, lr_coeff`
  - 계수 히스토리(epoch 단위): `b, s_r, alpha_L, alpha_R`
  - 파라미터: `TrainConfig` 전체 dump → `config.json`
  - 아티팩트: `best.pt` (state_dict + optimizer_state + val_data + coefficients + config), `history.json`

## 학습 설정
| 항목 | 값 | 비고 |
|------|-----|------|
| Optimizer | AdamW | MLP / coeff 파라미터 그룹 분리 |
| MLP Learning Rate | 1e-3 | `mlp_lr` |
| Coeff Learning Rate | 3e-4 | `mlp_lr × coeff_lr_mult (=0.3)` |
| Weight Decay | 1e-4 (MLP only) | coeff group은 0 (prior loss 로 대체) |
| LR Scheduler | CosineAnnealingLR | `T_max=epochs`, `eta_min = mlp_lr × 0.01` |
| Batch Size | 64 | 요구 기본값 |
| Epochs (max) | 300 | |
| Early Stopping | patience=20 | monitor=`val_data` (pure data MSE) |
| Gradient Clipping | max_norm=1.0 | 전 파라미터 대상 |
| Mixed Precision | True (CUDA일 때 자동) | CPU는 자동 비활성 |
| Loss 가중치 | `w_data=1.0, w_stat=1.0, w_nh=0.1, w_coeff=0.01, w_mag=1e-3` | 02_model_design 기본값 그대로 (lambda_lat는 v2에서 제거됨) |

## 재현성 설정
- **Random Seed**: 42 (기본) — `random`, `numpy`, `torch`, `cuda` 전체 고정
- **PYTHONHASHSEED**: `os.environ["PYTHONHASHSEED"] = "42"` (set_seed에서 설정)
- **CUBLAS_WORKSPACE_CONFIG**: `:4096:8` (deterministic matmul)
- **torch.backends.cudnn.deterministic**: True
- **torch.backends.cudnn.benchmark**: False
- **환경 기록**: `config.json`에 torch 버전·device·AMP 여부 기록, `requirements.txt`는 Colab 노트북 셀 1에서 버전 출력

## 체크포인트 전략
- **저장 조건**: `val_data` 가 개선될 때마다 `best.pt` 덮어쓰기
- **저장 경로**: `{save_dir}/best.pt` (Colab에서 `/content/runs/pinn/best.pt`, Drive mount 시 `/content/drive/MyDrive/...`)
- **저장 내용**: `{epoch, model_state, optimizer_state, val_data, val_total, coefficients, config}`
- **모델 레지스트리**: 외부 서비스 미사용. Colab은 `files.download("best.pt")` 또는 Drive 복사 권장
- **재개**: `load_best(model, best_path)` 유틸 제공 (optimizer_state는 필요 시 직접 로드)

## 하이퍼파라미터 튜닝
- **도구**: 기본은 수동 grid (코드 내 `SUGGESTED_GRID`). Optuna wrapper 는 후속 옵션.
- **탐색 공간 제안** (02_model_design 참조):
  - `mlp_lr ∈ [1e-4, 5e-3]` (log-uniform)
  - `coeff_lr_mult ∈ [0.1, 1.0]` (log-uniform, 기본 0.3)
  - `hidden_dim ∈ {64, 96, 128}`
  - `dropout ∈ [0.0, 0.3]`
  - `w_nonholonomic ∈ [0.01, 1.0]` (log-uniform)
  - `w_coeff ∈ [1e-3, 0.1]` (log-uniform)
- **목적 함수**: minimize `val_data` (순수 예측 성능; physics 항 배제로 튜닝 편향 방지)
- **조기 종료**: 학습 자체에 early_stop_patience=20 존재. Optuna 적용 시 `MedianPruner` 권장.

## 학습 스크립트
구현: `_workspace/experiment_code/train_module.py`

핵심 공개 API:
```python
from train_module import TrainConfig, train_api, load_best, set_seed

cfg = TrainConfig(
    epochs=300, mlp_lr=1e-3, coeff_lr_mult=0.3, weight_decay=1e-4,
    grad_clip_max_norm=1.0, early_stop_patience=20,
    seed=42, amp=True, deterministic=True,
)
result = train_api(model, loss_fn, train_loader, val_loader, cfg,
                   save_dir="/content/runs/pinn")
# result = {"history": {...}, "best_path": ".../best.pt"}
load_best(model, result["best_path"])
```

루프 요약:
1. `set_seed(cfg.seed)` → device 결정(auto) → model/loss to(device)
2. `build_optimizer` (AdamW, 2 groups) + `build_scheduler` (Cosine)
3. AMP scaler (CUDA 일 때만)
4. epoch 루프:
   - `train_one_epoch`: 배치당 forward / (AMP unscale) / grad-clip / optimizer.step
   - `evaluate`: val 전체 에서 5개 loss 평균
   - scheduler.step()
   - `model.coefficient_values()` snapshot → 로거
   - `val_data` 개선 시 `best.pt` 저장
   - `EarlyStopping.step(val_data)` → 조기 종료
   - history.json 주기적 flush (크래시 내성)

## 인프라 요구사항
| 리소스 | 최소 | 권장 |
|--------|------|------|
| GPU | 없음 (CPU 학습 가능) | Colab T4/L4/A100 (AMP 혜택) |
| VRAM | N/A | 2 GB (배치 64, H=128 기준 여유) |
| RAM | 2 GB | 4 GB |
| Storage | 50 MB (체크포인트+로그) | 500 MB |

모델 크기 ≈ 50.7k params → 50 Hz 샘플링 데이터 1k–100k 행 기준 in-memory 로 충분.

## Colab 노트북
`_workspace/experiment_code/colab_notebook.ipynb` 8셀 구성:
1. 환경 체크 (`torch.__version__`, GPU 확인) + 최소 pip install
2. 코드 파일 로드 (Drive mount 또는 `files.upload()` 두 옵션 주석)
3. CSV 2개 업로드 → `upload_data_colab()`
4. `load_and_align → split_timeseries → fit_scaler → build_dataloaders` + stationary ratio 출력
5. 모델/loss/optimizer 빌드 + smoke forward
6. `train_api(...)` 호출 → history/best_path
7. matplotlib loss curves + coefficient trajectory
8. `load_best` → test 예측 + MSE/MAE 지표

## 에러 핸들링
- **GPU 미사용 환경**: `TrainConfig(device="cpu", amp=False)` 자동 설정. `resolve_device("auto")`가 fallback.
- **학습 발산 (NaN/Inf loss)**: AMP에서 자주 보고됨 → `GradScaler` 가 skip step. 재현되면 `amp=False` 또는 `grad_clip_max_norm=0.5`, `mlp_lr → 5e-4` 권장.
- **Stationary ratio < 1%**: `L_stat` 의 `denom.clamp_min(1e-6)` 으로 수치 보호. 경고 권장.
- **OOM**: batch_size=32 로 감소, AMP 유지, `num_workers=0`.

## 평가분석가 전달 사항
- **산출 경로**: `save_dir/best.pt` (모델 가중치 + 계수 최종값), `save_dir/history.json` (epoch 메트릭 + coefficient trajectory)
- **학습 곡선**: `history.json → epochs[*].train_data / val_data / nonholonomic / stationary / coeff / magnitude`
- **계수 trajectory**: `history.json → coefficients[*] = {epoch, b, s_r, alpha_L, alpha_R}`
- **Test 평가 절차**: `load_best(model, best_path)` → test_loader 순회 → `accumulate_trajectory` 기반 ATE/RPE/FPE 계산
- **Ablation 지원**: `LossConfig(w_nonholonomic=0, w_stationary=0, w_coeff=0)` 별도 학습 후 비교
- **재현**: 모든 run `seed=42` 기본. baseline(Identity/Linear) 은 `TrainConfig` 동일 설정으로 비교 실행

## 리뷰어에게
- 데이터 누수 없음: scaler는 train split 에서만 fit (데이터모듈 보장)
- Best 기준: `val_data` (순수 MSE) — physics 항 가중치 tuning 으로 인한 편향 제거
- `weight_decay=0` on coefficient group → L2 prior loss 와 중복 방지
- 모든 run 에 `config.json` 기록 → 재현 가능
- Smoke test 포함: `python train_module.py` 실행 시 5 epoch 합성 데이터 검증
