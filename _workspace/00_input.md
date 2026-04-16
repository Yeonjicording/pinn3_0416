# 00. 사용자 입력 정리

## 연구 제목
Inverse Physics-Informed Neural Networks (Inverse-PINN)을 활용한 자율이동로봇 Odometry Error 최소화 시스템 계수 역추정

## 문제 정의
- **유형**: 회귀 + 물리 제약 기반 역문제(Inverse Problem) — 시스템 계수 추정 + Odometry 보정
- **도메인**: 자율주행 / 이동로봇 / Dead Reckoning
- **타겟 플랫폼**: Agilex Scout (differential drive, non-holonomic)
- **시나리오**: GPS-denied 환경에서 외부 마커/비콘 없이 odometry 만으로 정밀 위치 추정

## 해결하고자 하는 문제
- 자율이동로봇은 내부 에러(baseline, wheel radius scale, wheel acceleration, 좌우 휠 비대칭)와 외부 에러(지면, 기후, 장애물)로 인해 누적 드리프트 발생
- 기존 EKF는 비선형·시변 bias 특성 수식화 한계, End-to-End RNN/LSTM은 대량 GT 필요 + 물리 법칙 무시 → 과적합
- 본 연구는 **내부 에러만** 타겟으로 Inverse-PINN을 사용하여 odometry의 보정 계수를 역추정

## 입력 데이터
- **파일 형식**: CSV 2개 (사용자 업로드)
- **컬럼**: `t, d_x, d_y, d_z, d_rolling, d_pitch, d_yaw` (동일 구조)
  - `t`: timestamp
  - `d_x/d_y/d_z`: 각 축 위치 증가분 (delta)
  - `d_rolling/d_pitch/d_yaw`: 각 축 자세 증가분
- **파일 1 (GT)**: Ground Truth (학습 정답)
- **파일 2 (Odom)**: 로봇 내부 odometry (입력)
- **전처리 요구사항**: 두 파일의 timestamp를 자동 정렬/매칭

## 모델 출력
- 보정된 `d_x_corr`, `d_yaw_corr` (진행방향 위치 증가분 + yaw 자세 증가분)
- 동시에 시스템 계수(baseline `b`, wheel radius scale `s_r`, 좌우 비대칭 `α_L, α_R` 등)를 학습 가능 파라미터로 추정

## Physics Loss 요구 조건
1. **정지 조건 (Stationary)**: 입력 odometry가 정지 상태면 예측된 `d_x_corr, d_yaw_corr ≈ 0`
2. **Non-holonomic 제약**: Scout 플랫폼의 differential drive 운동 방정식 준수 (횡방향 속도 = 0, `d_y_body ≈ 0`)
3. **Magnitude Regularization**: 보정값이 과도하게 커지지 않도록 규제

## 실행 환경
- **Google Colab** 실행 가능 코드
- 파일 업로드 UI 지원 (`google.colab.files.upload()`)
- GPU 선택적 사용 (CPU도 동작)

## 제외 사항
- 외부 에러(지면, 기후, 장애물)는 예측 불가 → 본 연구에서 고려하지 않음

## 목표
- Odometry-only trajectory 대비 보정 후 trajectory의 누적 에러(ATE, RPE, 최종 위치 오차) 감소
- 학습된 시스템 계수의 해석 가능성
- 적은 학습 데이터로도 물리적으로 타당한 해 도출
