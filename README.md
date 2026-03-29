# 2025 인하 인공지능 챌린지 — 언어 정보 기반 이미지 색채화

> **2025 Inha AI Challenge** | 주관: 인하대학교 인공지능융합연구센터

---

## 대회 개요

| 항목 | 내용 |
|------|------|
| **주제** | 언어 정보(캡션) 기반 흑백 이미지 색채화 |
| **플랫폼** | Dacon |
| **대회 기간** | 2025.06.23(월) 10:00 ~ 2025.07.31(목) 18:00 |
| **코드 제출** | 2025.08.01(금) 14:00 |
| **순위 발표** | 2025.08.13(수) 18:00 |
| **시상식** | 2025.08.19(화) 14:00 |

**평가 방법**: 생성된 색채화 이미지를 CLIP ViT-L-14 모델로 임베딩한 뒤, 텍스트 캡션과의 코사인 유사도를 기준으로 평가

---

## 접근 방법 (Solution Overview)

Stable Diffusion 2.1 기반의 **Multi-ControlNet + LoRA** 파이프라인을 구축하여, 텍스트 캡션 정보를 최대한 활용한 색채화를 수행합니다.

### 전체 파이프라인

```
흑백 이미지 + 캡션
        │
        ├─ Grayscale control image
        ├─ Canny edge map
        └─ Depth map
              │
     Multi-ControlNet (3개 조합)
     + LoRA fine-tuned UNet
     + DPM-Solver++ Scheduler
              │
        색채화 이미지 (RGB)
              │
     LAB 후처리 (L 채널 원본 교체)
              │
        최종 출력
```

---

## 학습 전략

### Phase 1 — UNet LoRA Fine-tuning (`train_sd21_unet_lora_e1.ipynb`)

- Base 모델: `stabilityai/stable-diffusion-2-1-base`
- 흑백 이미지 → 컬러 이미지 태스크로 SD 2.1 UNet을 **LoRA** (rank=4, alpha=4) 로 1 epoch fine-tuning
- LoRA 적용 대상: `to_k`, `to_q`, `to_v`, `to_out.0` (어텐션 레이어)
- 학습률: `1e-4`, 배치 크기: 8, 혼합 정밀도(fp16)
- 캡션 토큰 길이 75 이하로 필터링하여 학습 데이터 전처리

### Phase 2 — ControlNet with HSV + CLIP Loss (`train_sd21_controlnet_hsvclip_e3.ipynb`)

- lllyasviel/ControlNet 프레임워크 기반으로 SD 2.1용 ControlNet 학습
- `ControlLDM_HSV_CLIP` 커스텀 모델 클래스: 기존 `ControlLDM` 상속 후 손실 함수 확장

**복합 손실 함수:**

```
Loss = MSE(noise) + λ_hsv × HSV-L1 + λ_clip × (1 - CLIP cosine similarity)
```

| 손실 항목 | 가중치 | 설명 |
|-----------|--------|------|
| Noise MSE | 1 | 표준 확산 모델 denoising 손실 |
| HSV-L1 | λ=10 | 색상(H), 채도(S), 명도(V) 채널별 가중 L1 (H에 3배 가중) |
| CLIP Cosine | λ=5 | 생성 이미지와 텍스트 캡션 간 의미 정렬 |

- 학습률: `1e-5`, 배치 크기: 4, 3 epoch 학습
- PyTorch Lightning + ModelCheckpoint로 epoch 단위 가중치 저장

---

## 추론 파이프라인 (`inference.ipynb`)

### Multi-ControlNet 구성

| ControlNet | 조건 입력 | 가중치 |
|------------|----------|--------|
| 커스텀 (Phase 2) | Grayscale 이미지 | 0.7 |
| Canny edge | Canny 엣지 맵 | 0.5 |
| Depth | MiDaS 깊이 맵 | 0.3 |

### 프롬프트 처리

- 색상 관련 단어 자동 강조: `red` → `((red))` (최대 10개)
- 77 토큰 초과 시 **Compel** 라이브러리로 long prompt 처리
- Prefix 추가: `"vibrant natural colors, realistic lighting, balanced white-balance."`

### 추론 설정

```python
num_inference_steps = 72
guidance_scale      = 7.5
cfg_rescale         = 0.5
scheduler           = DPMSolverMultistepScheduler (dpmsolver++, karras sigmas)
```

### 후처리 — LAB 밝기 보정

원본 흑백 이미지의 밝기 정보를 보존하기 위해 **LAB 색 공간**에서 L 채널을 교체:

```python
color_lab[:, :, 0] = gray_lab[:, :, 0]  # 밝기(L)를 원본 그레이스케일로 교체
```

---

## 제출 형식

- 생성 이미지를 **CLIP ViT-L-14**로 임베딩 후 CSV로 저장 (`embed_submission.csv`)
- 이미지 파일 + CSV를 ZIP으로 압축하여 Dacon에 제출

---

## 프로젝트 구조

```
.
├── train_sd21_unet_lora_e1.ipynb          # Phase 1: LoRA fine-tuning
├── train_sd21_controlnet_hsvclip_e3.ipynb # Phase 2: ControlNet + HSV+CLIP loss
├── inference.ipynb                         # 최종 추론 및 제출 파일 생성
├── requirements.txt                        # 의존성 패키지
├── data/                                   # 학습/추론 데이터 (gitignore)
├── models/                                 # 모델 가중치 (gitignore)
└── submission/                             # 제출 파일
```

---

## 환경 설정

```bash
pip install -r requirements.txt
```

주요 의존성:

| 패키지 | 버전 |
|--------|------|
| torch | 2.6.0 |
| diffusers | 0.34.0 |
| transformers | 4.54.1 |
| open-clip-torch | 2.24.0 |
| controlnet-aux | 0.0.10 |
| kornia | 0.8.1 |
| compel | 2.1.1 |
| pytorch-lightning | 1.5.0 |

> GPU 환경(CUDA) 필수. 학습 및 추론은 Google Colab A100 환경에서 수행하였습니다.
