# 🌬️ AirLLM 전수조사 분석 리포트 (한국어)

> 작성: 카리나 (Claude Code) 💖 · 정리일: 2026-10-08
> 대상 저장소: **https://github.com/bmshin94/airllm**
> 원본(업스트림): **https://github.com/lyogavin/airllm** (Gavin Li, Apache-2.0)
> 분석 브랜치: `claude/elegant-darwin-wbjati`

---

## 📑 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [핵심 원리: 레이어 스트리밍](#2-핵심-원리-레이어-스트리밍)
3. [폴더 전수조사](#3-폴더-전수조사)
4. [코드에서 확인한 고급 기술 8가지](#4-코드에서-확인한-고급-기술-8가지)
5. [성능 표 / 솔직한 단점](#5-성능-표--솔직한-단점)
6. [쉬운 비유로 이해하기](#6-쉬운-비유로-이해하기)
7. [설치 및 사용법](#7-설치-및-사용법)
8. [Q&A: 플러그인? 스킬? MCP?](#8-qa-플러그인-스킬-mcp)
9. [Q&A: API 토큰 필요 여부](#9-qa-api-토큰-필요-여부)
10. [Q&A: AI 에이전트 구축에 도움이 되나](#10-qa-ai-에이전트-구축에-도움이-되나)
11. [Q&A: React / PHP로 만들 수 있나](#11-qa-react--php로-만들-수-있나)
12. [Q&A: 유튜브 강의 제작 가능성](#12-qa-유튜브-강의-제작-가능성)
13. [수익화 아이디어 8선](#13-수익화-아이디어-8선)
14. [3단 로켓 실행 전략](#14-3단-로켓-실행-전략)
15. [참고 링크 모음](#15-참고-링크-모음)

---

## 1. 프로젝트 개요

### 한 줄 요약

> **4GB VRAM 그래픽카드 1장으로 70B~2.8T 초거대 LLM을 돌리는 Python 라이브러리**

| 항목 | 내용 |
|---|---|
| 정체 | Python 라이브러리 (PyPI 패키지 `airllm`) |
| 패키지 버전 | `4.0.0` (`air_llm/setup.py`) |
| 라이선스 | **Apache-2.0** → 상업적 이용 가능 ✅ |
| 저장소 규모 | 56 커밋, 약 11MB (코드 위주) |
| 특징 | **양자화·증류·프루닝 없이** VRAM 요구량만 줄임 (정확도 손실 0) |
| 개발자 | Gavin Li (`lyogavin`) |

### 지원 모델

Llama (2/3/3.1/3.3/4) · Qwen (1/2/2.5/3/3.5/3.8, MoE·Flash-Next·FP8·VL 포함) ·
DeepSeek (V2/V3/R1) · Mistral & Mixtral · Phi · Gemma · ChatGLM · Baichuan ·
InternLM · Yi · Kimi K3 — 그리고 transformers가 지원하는 **대부분의 신규 모델**

---

## 2. 핵심 원리: 레이어 스트리밍

### 기존 방식의 문제

Transformer 추론은 보통 **모델 전체를 VRAM에 상주**시킨다.

- Llama 70B (bf16) → 약 140GB → A100 80GB × 2장
- DeepSeek-V3 671B → 약 1.3TB → H100 × 16장

### AirLLM의 발상

```
입력 → [임베딩] → [레이어 0] → [레이어 1] → ... → [레이어 79] → [norm] → [lm_head] → 출력
```

**레이어 0의 계산이 끝나면 레이어 0의 가중치는 더 이상 필요 없다.**

```
1. 레이어 N 가중치를 디스크 → GPU 로드
2. forward 계산 실행
3. 가중치를 GPU에서 제거 (meta 디바이스로 반환)
4. 레이어 N+1 반복
```

### 결론

> **필요 VRAM = 모델 전체 크기 ❌ → 가장 큰 레이어 1개 크기 ⭕**

---

## 3. 폴더 전수조사

저장소는 사실상 **두 개의 프로젝트**가 합쳐진 구조다.

### A. `air_llm/` — 메인 프로젝트 (AirLLM 본체)

```
air_llm/
├── setup.py                       # PyPI 패키지 정의 (v4.0.0)
├── inference_example.py           # 가장 단순한 추론 예제
├── airllm/
│   ├── __init__.py          (48)  # 공개 API, 옵셔널 임포트 방어 처리
│   ├── auto_model.py        (63)  # ⭐ AutoModel — 아키텍처 자동 감지/디스패치
│   ├── airllm_base.py      (920)  # 🔥 심장부. 스트리밍 전체 로직
│   ├── utils.py            (870)  # 🔥 체크포인트 분할/저장/mmap/압축
│   ├── airllm_lora.py      (562)  # ⭐ 스트리밍 LoRA 학습기
│   ├── lora_linear.py      (201)  # 자체 구현 LoRA (PEFT 미사용)
│   ├── chunked_ce.py       (144)  # 청크 단위 Cross-Entropy
│   ├── lora_data.py        (151)  # JSONL/JSON/TXT 데이터 로더
│   ├── profiler.py          (31)  # 레이어별 시간/메모리 측정
│   ├── airllm_llama_mlx.py (436)  # macOS(Apple Silicon) MLX 전용 경로
│   ├── airllm_kimi_k3.py          # Kimi K3 2.8T 전용 (per-expert 스트리밍)
│   ├── airllm_qwen4_exp.py        # Qwen3.8-Flash-Next 전용 (mmap PLE)
│   ├── airllm_qwen3_5.py          # Qwen3.8-27B VL 전용
│   ├── airllm_{chatglm,qwen,qwen2,baichuan,internlm,mistral,mixtral}.py
│   ├── tokenization_baichuan.py   # Baichuan 전용 토크나이저
│   └── persist/                   # 저장 백엔드 (safetensors / MLX)
├── examples/                      # 노트북 3개 + LoRA 학습 스크립트 2개
│   ├── run_all_types_of_models.ipynb
│   ├── run_llama3.1_405B.ipynb
│   ├── run_on_macos.ipynb
│   ├── train_qwen38_lora.py
│   ├── train_qwen38_flash_next_lora.py
│   └── sft_example.jsonl          # 2줄짜리 학습 데이터 샘플
└── tests/                         # pytest 11개 파일 + 노트북 테스트 6개
```

### B. 레거시 "Anima" 프로젝트 (AirLLM 사용엔 불필요)

| 폴더 | 내용 |
|---|---|
| `anima_100k/` | Llama2-7B를 **100K 컨텍스트**로 확장. FlashAttention2 + XEntropy로 596GB → 782MB 메모리 최적화 |
| `training/` | **Anima 33B** — 최초의 QLoRA 기반 오픈소스 중국어 LLM 학습 코드 |
| `rlhf/` | QLoRA + **DPO** 저비용 RLHF 구현 (GPU 1대로 33B RLHF) |
| `eval/` | ELO 토너먼트 방식 모델 평가 노트북 |
| `data/` | GPT-4로 번역한 Vicuna 평가셋 |
| `scripts/` | 스타 히스토리 차트 생성, 데이터셋 길이 측정 |

### C. 설정 / 메타 파일

| 파일 | 설명 |
|---|---|
| `CLAUDE.md`, `GEMINI.md` | **이 포크에서 직접 추가한 파일** (최신 커밋 `8f8a3c5 "Add files via upload"`). AI 어시스턴트 페르소나("카리나") 설정. 원본 저장소에는 없음 |
| `.github/workflows/release.yml` | GitHub Release 발행 시 PyPI 자동 배포. **OIDC Trusted Publishing** 사용 (API 토큰 미저장, 보안 양호). 태그와 `setup.py` 버전 불일치 시 실패 가드 있음 |
| `.github/workflows/star-history.yml` | 매일 06:00 UTC 스타 히스토리 PNG 재생성 → 커밋 (56커밋 중 대부분이 이 자동 커밋) |
| `funding.json`, `.github/FUNDING.yml` | 후원 설정 (GitHub Sponsors / Buy Me a Coffee) |
| `requirements.txt` | ⚠️ **레거시 Anima 전용** (2023년 버전 핀). AirLLM 의존성은 `air_llm/setup.py` 참조 |
| `LICENSE` | Apache 2.0 |

---

## 4. 코드에서 확인한 고급 기술 8가지

### 1️⃣ `meta` 디바이스 트릭 — `airllm_base.py`

```python
with init_empty_weights(include_buffers=False):
    model = cls.from_config(self.config)
```

PyTorch `meta` 디바이스 = **모양만 있고 데이터가 없는 텐서**.
671B 모델의 *구조*를 메모리 0 byte로 만들고, 가중치만 필요할 때 끼워 넣는다.
`include_buffers=False`인 이유: rotary `inv_freq` 같은 비영구 버퍼는 체크포인트에 없으므로 실제로 계산되어야 함.

### 2️⃣ Forward Hook 기반 자동화

```python
module.register_forward_pre_hook(self._pre_hook)   # 실행 직전 → 디스크에서 로드
module.register_forward_hook(self._post_hook)      # 실행 직후 → meta로 반환
```

**핵심 설계:** forward 연산 자체는 **transformers가 그대로 담당**하고, AirLLM은 가중치 수급만 관리한다.
→ 신규 아키텍처가 나와도 transformers가 지원하면 **코드 수정 없이 동작** (`auto_model.py` 주석에 명시).

### 3️⃣ 프리페칭 (Prefetching)

```python
self._executor = ThreadPoolExecutor(max_workers=1)
# 레이어 N 계산 중 백그라운드 스레드가 레이어 N+1을 미리 읽는다
```

디스크 I/O와 GPU 연산을 중첩 → **약 10% 속도 향상**.
`pin_memory()`로 페이지 고정 메모리를 쓰되, **2GB 이하 레이어만** 적용
(`max_pinned_layer_bytes = 2 * 1024**3`) — MoE의 17GB 레이어까지 고정하면 RAM 34GB가 잠기므로 의도적으로 제외.

### 4️⃣ MoE 전문가별 스트리밍 — `_setup_expert_streaming()`

Kimi K3는 레이어당 전문가가 **896개**인데 토큰 하나는 **16개만** 라우팅한다.

```
레이어 전체 전문가 (전개 시) ≈ 55GB
토큰 1개가 실제 사용 ≈ 1GB
```

→ 각 전문가 모듈에 **개별 forward hook**을 달아 라우팅된 전문가만 로드.
2.8T 모델이 3.72GB에 들어가는 비밀.
(조건: `expert_prefix` 지정 + safetensors 포맷)

### 5️⃣ mmap 임베딩 — Qwen3.8-Flash-Next

Flash-Next의 n-gram(PLE) 테이블은 **bf16 약 102GB**.

→ `MmapEmbedding` 클래스로 **디스크 파일을 메모리 맵핑**, 필요한 행만 직접 gather.
→ **64GB RAM 머신에서도 실행 가능** (RAM/VRAM에 102GB를 올리지 않음).
`_force_meta_embeddings()`로 accelerate가 191GB fp32를 CPU에 먼저 할당하는 것도 차단.

### 6️⃣ 모델 압축 = 최대 3배 속도 향상

```python
model = AutoModel.from_pretrained("...", compression='4bit')  # or '8bit'
```

**발상의 전환:** AirLLM의 병목은 연산이 아니라 **디스크 읽기 대역폭**이다.
→ 활성값은 그대로 두고 **가중치만** 블록 단위 양자화 → 파일 크기 축소 → 읽기 속도 향상
→ 최대 **3배 속도업, 정확도 손실 거의 0**
(근거 논문: https://arxiv.org/abs/2212.09720)
⚠️ 압축과 프리페칭은 동시 사용 불가 (코드에서 프리페칭 자동 비활성화)

### 7️⃣ FP8 / MXFP4 압축 텐서 지원

```python
def _should_load_verbatim(self, param_name, value):
    if not value.is_floating_point(): return True   # 패킹된 4bit, zero_point, g_idx
    if value.element_size() == 1:     return True   # fp8 e4m3/e5m2, MXFP4 e8m0 스케일
    return param_name.endswith(self._QUANT_COMPANION_SUFFIXES)
```

양자화 가중치를 **압축 상태로 PCIe 전송 → GPU에서 압축 해제**.
MXFP4는 전송량이 **1/4**. 또한 compressed-tensors가 등록하는 전체 모델 압축해제 훅을
`init_model()`에서 **제거**해서 per-expert 스트리밍이 무력화되는 것을 막는다.

### 8️⃣ 스트리밍 LoRA 학습 — `airllm_lora.py` (2026/09 신기능)

125B 모델을 **6GB RTX 3060 Ti**로 파인튜닝.

```python
class _StreamedModule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden):
        h_cpu = hidden.detach().cpu()                   # hidden state를 CPU 보관
        with torch.no_grad():
            out = trainer._run_streamed(kind, h_cpu, extras, grad=False)
        ctx.save_for_backward(h_cpu)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        h = ctx.saved_tensors[0].to(trainer.device).requires_grad_(True)
        out = trainer._run_streamed(ctx.kind, h, ctx.extras, grad=True)  # 해당 레이어만 재계산
        out.backward(g)
        ...
```

- 동결 베이스 가중치는 **디스크에서 한 레이어씩** 스트리밍
- **hidden state는 CPU에 보관** → 64개 레이어 그래프가 VRAM을 점유하지 않음
- **LoRA A/B + Adam만 GPU 상주**
- backward 때 **해당 레이어만 재계산** (gradient checkpointing의 극단 버전)
- `chunked_ce.py`가 `[N, 248320]` 로짓 전체 생성을 회피 (vocab을 청크 순회하며 log-sum-exp 누적)
- **PEFT / HF Trainer 미사용** — 둘 다 베이스 가중치가 실제로 메모리에 있다고 가정하기 때문
- Dropout은 0 (backward 재계산이 forward와 일치해야 함)

### 보너스: 호환성 방어 코드들

- `restore_relocated_transformers_symbols()` — transformers 5.0이 옮긴 심볼을 구 위치에 재노출 (구버전 대상으로 쓰인 remote code 구제)
- `_auto_model_classes()` — VL 체크포인트가 text-only 클래스에 잘못 매핑되는 문제 해결 (아키텍처명으로 팩토리 우선순위 조정)
- `_patch_device_property()` — 파라미터가 meta에 있어도 모델이 cuda를 보고하게 패치 + `GenerationMixin` 재주입
- `_adopt_checkpoint_shape()` — 모델 클래스가 config로 만든 모양과 체크포인트가 다를 때 체크포인트를 신뢰 (Kimi K3 `A_log`)
- `_wrap_forward_int64_scatter()` — transformers qwen4_exp의 int32 scatter 인덱스 버그 우회

---

## 5. 성능 표 / 솔직한 단점

### VRAM 요구량 (README 실측 기준)

| 모델 | 총 파라미터 | 필요 VRAM | 측정 환경 |
|---|---|---|---|
| Qwen3 / Mistral / Phi | 약 8B | ~1–2 GB | - |
| Qwen3-30B / Mixtral (MoE) | 30–47B | ~1–3 GB | - |
| Qwen3.8-27B (dense VL) | 27B | **3.33 GB** | RTX 3090 |
| Qwen3.8-Flash-Next (MoE+PLE) | ~180B | **5.95 GB** | RTX 4090 |
| Qwen3-235B (MoE) | 235B | ~3 GB | - |
| Llama 3.x 70B (풀정밀도) | 70B | ~4 GB | - |
| Llama 3.1 405B | 405B | ~8 GB | - |
| DeepSeek-V3 | **671B** | ~12 GB | - |
| Kimi K3 | **2.8T** | **3.72 GB** | RTX 6000 Ada |

### 학습(LoRA) VRAM

| 모델 | VRAM | 조건 |
|---|---|---|
| Qwen3.8-27B | **~2 GB** | seq 512 |
| Qwen3.8-Flash-Next (125B) | **6 GB 미만** | RTX 3060 Ti, seq 512 |

### ⚠️ 솔직한 단점 (반드시 알아야 할 것)

| 단점 | 설명 |
|---|---|
| 🐌 **매우 느림** | 토큰 1개 생성마다 **모델 전체를 디스크에서 읽음**. 70B면 토큰당 약 140GB 읽기. NVMe(3GB/s)로도 토큰당 수십 초. vLLM과 비교 불가 |
| 💾 **디스크 폭식** | 원본 + 분할본 → **모델 크기의 2배**. Flash-Next는 약 360GB. `delete_original=True`로 절반 회수 |
| ⏳ **첫 실행 지연** | 체크포인트를 레이어별로 쪼개 저장하는 과정이 오래 걸림 (1회성) |
| 🔌 **NVMe SSD 필수** | HDD는 현실적으로 불가능 |
| 📦 **배치/동시요청 부적합** | 프로덕션 API 서빙에 완전히 부적합 |

### 핵심 트레이드오프

> **AirLLM = "속도를 팔아 VRAM을 사는" 장치**
> 실시간 챗봇 ❌ / 배치·연구·보안·개인실험 ⭕

---

## 6. 쉬운 비유로 이해하기

### 📚 비유 1: 80권짜리 백과사전

- **기존 방식**: 책상에 80권을 전부 펼쳐야 읽을 수 있음 → 거대한 책상(140GB VRAM) 필요
- **AirLLM**: 1권 읽고 덮어서 꽂고, 2권 꺼내 읽고 반복 → 책상엔 항상 1권만(4GB VRAM)
- 결과: 책상은 작아졌지만, 책 꺼내고 넣는 시간 때문에 느려짐

### 🍱 비유 2: 뷔페 vs 코스 요리

- 일반 LLM = 뷔페 (음식 전부를 한 테이블에) → 거대한 홀 필요
- AirLLM = 코스 요리 (한 접시씩 내고 치움) → 1인 테이블로 충분
- **맛(모델 성능)은 완전히 동일** — 양자화처럼 묽게 만드는 게 아니라 서빙 방식만 변경

### 🏥 비유 3: MoE 전문가 스트리밍

Kimi K3의 한 레이어 = 의사 896명이 대기하는 병원

- 기존: 환자 1명 와도 **896명 전원 출근** → 병원 터짐 (55GB)
- AirLLM: "이 환자는 피부과+정형외과만" → **16명만 호출** (1GB)

### 🎬 비유 4: `meta` 디바이스 = 영화 세트장

- 겉모습(모델 구조)은 완벽하게 다 있음
- 안은 텅 비어있음 (메모리 0 byte)
- 카메라가 1층 찍을 때만 1층에 진짜 가구를 들임 → 2층 찍을 때 1층 가구를 뺌

### ⚙️ 실제 동작 순서

```
[1단계] 처음 한 번만 (느림)
  허깅페이스 다운로드
    → 레이어별 조각 파일로 분할 저장
      (model.layers.0.safetensors, model.layers.1.safetensors, ...)

[2단계] 매 토큰마다 반복
  레이어0 읽기 → GPU → 계산 → 삭제
  레이어1 읽기 → GPU → 계산 → 삭제
  ... (80회) ...
  → 토큰 1개 완성 → 다음 토큰 위해 또 80회
```

---

## 7. 설치 및 사용법

### 기본 설치

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# PyTorch (CUDA 12.x) — 반드시 GPU 빌드
pip install torch --index-url https://download.pytorch.org/whl/cu124

# AirLLM
pip install airllm
```

자동 설치되는 의존성 (`air_llm/setup.py`):
`tqdm`, `torch>=2.4`, `transformers>=4.49,<6`, `accelerate>=1.0`,
`safetensors`, `huggingface-hub`, `scipy`, `sentencepiece`

### 선택 설치

```bash
pip install -U bitsandbytes       # 4bit/8bit 압축 (최대 3배 속도업)
```

### 가장 기본 코드

```python
from airllm import AutoModel

MAX_LENGTH = 128
model = AutoModel.from_pretrained("Qwen/Qwen3-32B")

input_text = ['What is the capital of United States?']

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False,
)

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(generation_output.sequences[0]))
```

### 모델 교체는 1줄

```python
model = AutoModel.from_pretrained("Qwen/Qwen3.8-27B")          # 27B  → 3.33GB
model = AutoModel.from_pretrained("Qwen/Qwen3.8-Flash-Next")   # 125B → 5.95GB
model = AutoModel.from_pretrained("Qwen/Qwen3-235B-A22B")      # 235B → ~3GB
model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-V3")   # 671B → ~12GB
```

### 설정 옵션 전체

| 옵션 | 설명 | 추천 |
|---|---|---|
| `compression` | `'4bit'` / `'8bit'` 블록 양자화 | `'4bit'` |
| `delete_original` | 분할 후 원본 삭제 (디스크 절반 절약) | 용량 부족 시 `True` |
| `layer_shards_saving_path` | 분할본 저장 위치 (빠른 NVMe로) | `/mnt/nvme/shards` |
| `prefetching` | 로딩/연산 중첩 (기본 ON) | `True` |
| `profiling_mode` | 레이어별 시간 측정 출력 | 튜닝 시 `True` |
| `hf_token` | gated 모델용 HF 토큰 | 필요 시 |
| `max_seq_len` | 최대 시퀀스 길이 (기본 512) | 용도에 맞게 |
| `device` | 실행 디바이스 (기본 `cuda:0`) | - |
| `dtype` | 런타임 dtype (기본은 모델 config 값, 보통 bf16) | 기본 유지 권장 |

```python
model = AutoModel.from_pretrained(
    "meta-llama/Meta-Llama-3-70B",
    compression='4bit',
    delete_original=True,
    layer_shards_saving_path='/mnt/nvme/shards',
    profiling_mode=True,
)
```

> ⚠️ `dtype`에 fp16을 강제하지 말 것. 깊은 모델(Qwen3-235B의 94 레이어 등)에서 inf/NaN 오버플로가 발생해
> 출력이 조용히 깨진다. 코드가 기본적으로 bf16을 선택한다.

### macOS (Apple Silicon)

```bash
pip install mlx
pip install airllm
```
코드는 리눅스와 동일. `__init__.py`가 darwin을 감지해 자동으로 `AirLLMLlamaMlx` 경로로 전환.
⚠️ Apple Silicon(M1+)만 지원, Intel Mac은 불가.

### 특수 모델 추가 요구사항

**Kimi K3 (2.8T)**
```bash
pip install compressed-tensors flash-attn   # 모델 코드가 flash attention 강제
pip install "transformers==4.56.*"          # 5.x에서 remote code 로드 실패
# torch는 CUDA 12 빌드 필수 (CUDA 13용 flash-attn 휠 없음)
```

**Qwen3.8-Flash-Next (125B)**
```bash
pip install git+https://github.com/huggingface/transformers.git  # in-tree qwen4_exp 필요
# 체크포인트 디스크 약 360GB, delete_original=True로 회수
# 호스트 RAM 64GB 이상 권장
```

**Qwen3.8-27B**: `transformers` 5.8+ 필요

### 하드웨어 체크리스트

```
GPU         : NVIDIA, VRAM 4GB 이상 (CPU 추론도 지원하지만 더 느림)
저장공간     : 모델 크기의 2배 (70B → 약 280GB), 반드시 NVMe SSD
RAM         : 32GB 권장 (Flash-Next는 64GB)
```

### LoRA 학습

데이터 형식 (`.jsonl`, 한 줄에 JSON 객체 하나):

```jsonl
{"text": "문서 전체에 대해 next-token 학습"}
{"prompt": "질문?", "completion": "답변"}
{"instruction": "번역해", "input": "bonjour", "output": "hello"}
```
(`prompt`/`completion` 형식은 **completion 부분에만 loss 적용**)

```bash
python air_llm/examples/train_qwen38_flash_next_lora.py \
  --data my_data.jsonl \
  --seq-len 512 \
  --epochs 1 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lr 1e-4 \
  --save-adapter qwen38-flash-next-lora.pt

# 27B 모델용
python air_llm/examples/train_qwen38_lora.py --data my_data.jsonl --save-adapter out.pt
```

`--steps 5`로 스모크 테스트부터 권장. `--data` 생략 시 내장 스니펫으로 오버핏 테스트.

Python API:

```python
from airllm import AirLLMLoRAQwen4Exp

trainer = AirLLMLoRAQwen4Exp(
    "Qwen/Qwen3.8-Flash-Next",
    max_seq_len=512,
    lora_r=16,
    delete_original=True,
)
tok = trainer.tokenizer
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

encoded = tok("학습 텍스트", return_tensors="pt", truncation=True, max_length=512)
loss = trainer.train_step(encoded["input_ids"].cuda(),
                          attention_mask=encoded.get("attention_mask"))
trainer.save_adapter("qwen38-flash-next-lora.pt")
```

### 자주 나는 에러 (README FAQ)

| 에러 | 원인 / 해결 |
|---|---|
| `MetadataIncompleteBuffer` | **디스크 공간 부족**. HF 캐시 정리 후 재시도 |
| `ValueError: max() arg is an empty sequence` | QWen/ChatGLM을 Llama2 클래스로 로드함 → `AutoModel` 사용 |
| `401 ... Repo model is gated` | HF 토큰 필요 → `hf_token=` 전달 |
| `Asking to pad but tokenizer has no padding token` | `padding=False` 또는 pad token 설정 |

---

## 8. Q&A: 플러그인? 스킬? MCP?

### 정답: 셋 다 아님 → **그냥 Python 라이브러리 (pip 패키지)**

| 구분 | 정체 | AirLLM? |
|---|---|---|
| 플러그인 | Claude Code / VSCode 기능 확장 모듈 | ❌ |
| 스킬 | AI에게 작업 방법을 알려주는 지침서(`SKILL.md`) | ❌ |
| MCP | AI가 외부 도구를 쓰게 하는 서버 프로토콜 | ❌ |
| **Python 라이브러리** | `import`해서 쓰는 코드 패키지 | ✅ |

### 계층 구조

```
┌──────────────────────────────────────┐
│  Claude Code / Cursor (AI 코딩 도구)  │ ← 플러그인/스킬/MCP가 사는 곳
├──────────────────────────────────────┤
│  내 애플리케이션 (FastAPI, Django...)  │
├──────────────────────────────────────┤
│  ★ AirLLM  (추론 엔진)                │
├──────────────────────────────────────┤
│  transformers / PyTorch               │
├──────────────────────────────────────┤
│  CUDA / GPU 드라이버                   │
└──────────────────────────────────────┘
```

### 같은 급의 경쟁/대안 라이브러리

| 도구 | 특징 |
|---|---|
| Ollama | 쉬운 로컬 실행, 빠름, 양자화 기반 |
| vLLM | 프로덕션 서빙 최강, 빠름, VRAM 많이 필요 |
| llama.cpp | CPU/GGUF 양자화, 가볍고 빠름 |
| **AirLLM** | **VRAM 최소화 극단 특화**, 느림 |

### 💡 기회 포인트

> **AirLLM을 감싼 MCP 서버는 아직 아무도 만들지 않았다.**

```python
# airllm-mcp-server (직접 만들 수 있는 아이템)
@mcp.tool()
def ask_local_giant_model(prompt: str) -> str:
    """로컬 DeepSeek-V3 671B에게 질문 (오프라인, 완전 프라이버시)"""
    return airllm_model.generate(prompt)
```

---

## 9. Q&A: API 토큰 필요 여부

### 결론: 거의 필요 없음. 완전 무료 + 오프라인

### 토큰이 필요 없는 경우 (대부분)

```python
model = AutoModel.from_pretrained("Qwen/Qwen3-32B")              # 토큰 X
model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-V3")     # 토큰 X
model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")  # 토큰 X
```

- OpenAI 토큰 ❌ / Anthropic 토큰 ❌ / AirLLM 자체 라이선스 키 ❌
- 인터넷은 **모델 다운로드 시에만** 필요 → 이후 **완전 오프라인 실행 가능**

### 토큰이 필요한 경우 (딱 하나)

허깅페이스 **Gated 모델** (사용 신청 승인 필요: Llama 계열 등)

```python
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", hf_token='hf_xxxxx')
```

```bash
huggingface-cli login     # 또는
export HF_TOKEN=hf_xxxxx
```

발급 절차 (전부 무료): huggingface.co 가입 → 모델 페이지에서 약관 동의 →
Settings → Access Tokens → New token (`read` 권한)

### 실제 비용

| 항목 | 비용 |
|---|---|
| AirLLM 라이브러리 | 0원 (Apache 2.0) |
| 모델 가중치 | 0원 (오픈소스 모델) |
| API 호출료 | 0원 (로컬 실행) |
| **실제 지출** | **전기값 + SSD 용량** |

---

## 10. Q&A: AI 에이전트 구축에 도움이 되나

### 결론: "반은 맞고 반은 틀리다"

### ❌ 부적합한 에이전트

```
실시간 대화형 에이전트
  → 에이전트는 한 작업에 LLM을 5~20회 호출 (ReAct 루프)
  → AirLLM은 1회 호출에 수십 초~수 분
  → 5회 호출 = 몇 분~몇십 분 대기

다중 사용자 동시 서비스 / 툴 호출이 많은 에이전트
  → 왕복이 많을수록 비현실적
```

### ✅ 최적인 에이전트

#### 1. 야간 배치 에이전트 ★★★★★

```python
for doc in 문서_10000개:
    분류 = airllm_671B.분석(doc)      # 느려도 무관 (수면 중 실행)
    DB.저장(분류)
```
"속도는 안 중요하고 품질이 중요한" 모든 작업.

#### 2. 프라이버시 필수 에이전트 ★★★★★

의료기록 / 법률문서 / 고객 개인정보 / 사내 기밀 →
클라우드 API 금지 분야. 기존엔 작은 7B 모델밖에 못 썼으나 671B 품질 확보.

#### 3. 교사 모델 (Teacher Model) ★★★★★

```
AirLLM으로 671B 고품질 학습데이터 10만 건 생성 (오프라인, 느리게)
      ↓
그 데이터로 작은 7B 모델 학습 (지식 증류)
      ↓
실제 서비스는 빠른 7B로 운영
```
실무에서 가장 유용한 패턴.

#### 4. 하이브리드 아키텍처 (가장 현실적)

```
사용자 질문
   ↓
[작은 빠른 모델 7B] ← 90%는 즉시 응답
   ↓ (어려운 질문)
[작업 큐 등록] → "분석 중, 5분 후 알려드립니다"
   ↓
[AirLLM 671B 백그라운드 처리]
   ↓
이메일 / 푸시 알림으로 결과 전송
```

#### 5. 학습·실험용 ★★★★★

에이전트 공부의 최대 장벽은 API 비용. AirLLM은 무한 호출이 공짜라서
루프 설계/프롬프트 실험을 마음껏 할 수 있다.

### 적합도 요약

| 에이전트 유형 | 적합도 |
|---|---|
| 실시간 챗봇 에이전트 | ★☆☆☆☆ |
| 툴 호출 많은 ReAct 에이전트 | ★★☆☆☆ |
| 야간 배치 에이전트 | ★★★★★ |
| 프라이버시 에이전트 | ★★★★★ |
| 데이터 생성 / 교사 모델 | ★★★★★ |
| 학습·실험용 | ★★★★★ |

> **"에이전트의 두뇌"로는 느리지만, "에이전트의 연구실·공장"으로는 최고.**

---

## 11. Q&A: React / PHP로 만들 수 있나

### 결론: AirLLM 자체의 포팅은 ❌, 그것을 쓰는 서비스는 ⭕

### 왜 포팅이 불가능한가

| AirLLM이 필요한 것 | React/PHP에 있나 |
|---|---|
| `torch` 텐서 연산 | ❌ |
| CUDA 커널 / GPU 직접 제어 | ❌ |
| `meta` 디바이스 (PyTorch 고유) | ❌ |
| `register_forward_hook` | ❌ |
| `torch.autograd.Function` 커스텀 | ❌ |
| safetensors / mmap 바이너리 처리 | 가능하지만 비현실적 |
| `transformers` 모델 구현체 수백 개 | ❌ |

→ PyTorch 생태계 전체를 재구현하는 일. 사실상 불가능.
(`transformers.js`, `onnxruntime-web`은 **작은 모델 전용**이며 접근 방식이 전혀 다름)

### 정답 구조 (적극 권장)

```
┌─────────────────────────────────────┐
│  프론트엔드                           │
│  React / Next.js / Vue / PHP(Laravel)│
└──────────────┬──────────────────────┘
               │ REST API / WebSocket / SSE
               ↓
┌─────────────────────────────────────┐
│  백엔드 (Python, 얇게)                │
│  FastAPI + AirLLM                    │
│  + Celery/Redis (작업 큐) ★ 필수      │
└──────────────┬──────────────────────┘
               ↓
            GPU (로컬/서버)
```

### Python 백엔드 예시

```python
# server.py — FastAPI + AirLLM + 작업 큐
from fastapi import FastAPI, BackgroundTasks
from airllm import AutoModel
import uuid

app = FastAPI()
model = AutoModel.from_pretrained("Qwen/Qwen3-32B", compression='4bit')
jobs = {}   # 실제 운영에서는 Redis 사용

def run_inference(job_id: str, prompt: str):
    tokens = model.tokenizer(prompt, return_tensors="pt",
                             return_attention_mask=False)
    out = model.generate(tokens['input_ids'].cuda(),
                         max_new_tokens=100, use_cache=True,
                         return_dict_in_generate=True)
    jobs[job_id] = {"status": "done",
                    "result": model.tokenizer.decode(out.sequences[0])}

@app.post("/api/generate")
def generate(prompt: str, bg: BackgroundTasks):
    """핵심: 즉시 응답하지 않고 작업 ID를 반환"""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    bg.add_task(run_inference, job_id, prompt)
    return {"job_id": job_id, "status": "pending"}

@app.get("/api/result/{job_id}")
def result(job_id: str):
    return jobs.get(job_id, {"status": "not_found"})
```

### React 프론트엔드 예시 (폴링)

```jsx
import { useState } from 'react';

export default function AirLLMChat() {
  const [status, setStatus] = useState('idle');
  const [answer, setAnswer] = useState('');

  const ask = async (prompt) => {
    setStatus('생각하는 중... (최대 몇 분 소요)');

    const { job_id } = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    }).then(r => r.json());

    const timer = setInterval(async () => {
      const r = await fetch(`/api/result/${job_id}`).then(r => r.json());
      if (r.status === 'done') {
        clearInterval(timer);
        setAnswer(r.result);
        setStatus('완료');
      }
    }, 3000);
  };

  return (
    <div>
      <button onClick={() => ask('안녕!')}>671B 모델에게 질문</button>
      <p>{status}</p>
      <pre>{answer}</pre>
    </div>
  );
}
```

### PHP (Laravel) 중계 예시

```php
// routes/api.php
Route::post('/generate', function (Request $r) {
    $res = Http::timeout(5)->post('http://127.0.0.1:8000/api/generate', [
        'prompt' => $r->prompt,
    ]);
    return $res->json();     // job_id 즉시 반환
});

Route::get('/result/{id}', fn($id) =>
    Http::get("http://127.0.0.1:8000/api/result/{$id}")->json()
);
```

### 설계 시 반드시 지킬 3가지

```
1. 동기 요청 금지 → 작업 큐(Celery/Redis/BullMQ) 필수 (HTTP 타임아웃)
2. 동시 요청 1개 제한 → GPU는 하나, 큐로 직렬 처리
3. UX 설계가 생명 → "5분 소요, 완료 시 알림" 솔직하게 안내 + 진행률 표시
```

### 역할 분담

| 담당 | 기술 | 역할 |
|---|---|---|
| 프론트엔드 개발자 | React / PHP / UI·UX | 화면, 작업 관리, 결과 표시, 알림 |
| Python | FastAPI + AirLLM | 추론 엔진 (얇은 래퍼, 200줄 수준) |

---

## 12. Q&A: 유튜브 강의 제작 가능성

### 결론: 매우 적합한 소재

조회수 3요소를 모두 충족:
1. **충격적 후킹** — "4GB 그래픽카드로 671B 모델 돌리기"
2. **공감되는 니즈** — "GPU 살 돈 없는 사람"
3. **경쟁자 없음** — 한국어 콘텐츠 사실상 제로

### 추천 시리즈 구성 (10편)

| # | 제목 | 길이 | 난이도 |
|---|---|---|---|
| 1 | "4GB 그래픽카드로 671B AI 돌리는 미친 방법" (충격 데모) | 8분 | 입문 |
| 2 | "어떻게 가능해? 레이어 스트리밍 원리 완전 해부" (애니메이션) | 12분 | 입문 |
| 3 | "따라하면 100% 되는 설치 가이드" (화면 녹화 풀버전) | 15분 | 실습 |
| 4 | "AirLLM vs Ollama vs vLLM — 실측 벤치마크 대결" | 15분 | 중급 |
| 5 | "4bit 압축으로 3배 빠르게 — 성능 튜닝 총정리" | 12분 | 중급 |
| 6 | "React + FastAPI로 나만의 AI 웹서비스 만들기" | 25분 | 실습 |
| 7 | "6GB GPU로 125B 모델 파인튜닝하기" (LoRA 실전) | 20분 | 고급 |
| 8 | "PyTorch 고수되기: meta 디바이스 + forward hook 코드 리딩" | 20분 | 고급 |
| 9 | "회사 기밀 데이터로 사내 AI 만들기 (온프레미스)" | 18분 | 실무 |
| 10 | "자는 동안 AI가 일하게 하기 — 배치 자동화" | 15분 | 실무 |

### 썸네일 / 제목 후킹 예시

```
"GPU 30만원 vs 3000만원, 결과는 똑같습니다"
"ChatGPT API 월 50만원 → 0원으로 만든 방법"
"내 노트북에서 2.8조 파라미터 AI가 돌아간다"
"671B 모델, 집에서 돌려봤습니다 (실패 아님)"
"AI 공부하는데 GPU 돈 없는 사람 보세요"
```

### 제작 팁

```
해야 할 것
  • 첫 15초에 "실제 동작 화면" 노출 (이탈 방어)
  • nvidia-smi로 VRAM 사용량 실시간 표시 → 신뢰도 상승
  • 레이어 스트리밍은 반드시 애니메이션/도식으로 설명
  • 느린 구간은 배속 + 자막으로 솔직하게 공개
  • GitHub 링크 / 설치 명령어는 고정 댓글 + 더보기란에

조심할 것
  • "빠르다"고 말하지 말 것 → 신뢰 상실. "느리지만 가능하다"로 포지셔닝
  • Apache 2.0이지만 원작자(lyogavin) 크레딧 반드시 표기
  • 환경 세팅 실패 댓글 대비 → 트러블슈팅 영상 별도 제작
  • 모델 다운로드 시간은 타임랩스 처리
```

### 유튜브 수익 구조

```
애드센스       : 구독 1만 기준 월 30~100만원
유료 강의 연계  : 인프런/클래스101 (가장 큰 수익)
하드웨어 제휴   : SSD/GPU 제휴 링크 (NVMe 필수라 궁합 좋음)
B2B 리드 유입   : 영상 → 기업 컨설팅 문의 (최고 단가)
전자책/노션     : "로컬 LLM 완전정복" 가이드 판매
```

---

## 13. 수익화 아이디어 8선

### 종합 순위표

| 순위 | 아이디어 | 난이도 | 초기비용 | 예상 수익 | 회수 기간 |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | **B2B 온프레미스 구축 컨설팅** | 상 | 낮음 | ★★★★★ | 3~6개월 |
| 2 | **유튜브 + 유료 강의** | 중 | 매우 낮음 | ★★★ | 2~4개월 |
| 3 | **합성 데이터 생성 서비스** | 중 | 중간 | ★★★★ | 3~6개월 |
| 4 | **파인튜닝 대행 서비스** | 상 | 중간 | ★★★★ | 4~8개월 |
| 5 | **오픈소스 생태계 기여 (MCP 등)** | 중 | 0원 | ★★ (간접) | 장기 |
| 6 | **배치 추론 SaaS** | 상 | 높음 | ★★★ | 6~12개월 |
| 7 | **전자책/템플릿/노션** | 하 | 0원 | ★ | 1개월 |
| 8 | **AI 박스 하드웨어 번들** | 상 | 매우 높음 | ★★★ | 1년+ |

---

### 1위. B2B 온프레미스 LLM 구축 컨설팅

**문제 상황 (실제 수요 다수)**

```
병원      : "환자 기록을 ChatGPT에 넣을 수 없음"
법무법인   : "계약서를 외부 API로 전송하면 징계 사유"
금융사     : "금융감독 규정상 클라우드 전송 금지"
대기업     : "사내 기밀문서 외부 유출 불가"
```

**기존 선택지**: A100 서버(3천만~1억, 예산 승인 불가) / 작은 7B 모델(품질 미달)

**AirLLM 솔루션**: RTX 4090 1장(300만원)으로 671B급 품질 + 완전 오프라인 + "계약서 검토 5분"은 허용 가능

**수익 모델 (한국 시장 기준)**

| 서비스 | 단가 |
|---|---|
| 초기 구축 (PoC) | 500만 ~ 2,000만원 |
| 모델 커스터마이징 (LoRA) | 300만 ~ 1,000만원 |
| 월 유지보수 | 50만 ~ 200만원/월 |
| 사내 교육 (반일/1일) | 100만 ~ 300만원 |

**실행 로드맵**

```
1단계 (1~2개월) 레퍼런스 확보
  • 무료/저가로 소규모 1건 수행
  • 데모 영상 + 벤치마크 자료 제작
  • "VRAM 4GB로 70B 동작" 증거 화면 녹화

2단계 (2~3개월) 영업 자료화
  • 산업별 유스케이스 문서 (의료/법률/금융/제조)
  • "클라우드 API vs 온프레미스" 비용 비교 계산기
  • 보안 규정 준수 체크리스트 (의료법, 전자금융감독규정 등)

3단계 (지속) 유입 채널
  • 유튜브/블로그 → 인바운드 문의 (2위 전략과 시너지)
  • LinkedIn, 로켓펀치, 원티드긱스
  • 중소기업 디지털 전환 지원사업 / AI 바우처 활용 ★
```

**주의**: 속도 한계를 처음부터 명시 ("실시간 챗봇 불가, 문서 분석 가능") /
SLA 조항 신중히 / 하이브리드 제안이 성공률 높음 / 데이터 처리 계약서 필수

---

### 2위. 유튜브 + 유료 강의 (진입 장벽 최저)

| 채널 | 가격대 | 예상 수익 |
|---|---|---|
| 유튜브 애드센스 | - | 월 30~100만원 (구독 1만) |
| **인프런 강의** | 5~15만원 | **수강생 200명 = 1,000~3,000만원** |
| Udemy (글로벌) | $30~100 | 글로벌 시장 |
| 전자책 (크몽/부크크) | 2~5만원 | 월 50~200만원 |
| 노션 템플릿/치트시트 | 1~3만원 | 월 10~50만원 |
| 제휴 마케팅 | - | NVMe SSD, GPU 제휴 |
| → B2B 리드 | - | 1위로 연결 (최고 단가) |

**유료 강의 커리큘럼 (총 10시간)**

```
Part 1. 왜 로컬 LLM인가 (1h)          — API 비용, 보안 규제, 장단점
Part 2. AirLLM 원리 완전 해부 (2h)     — ★차별화: 레이어 스트리밍, meta 디바이스,
                                        forward hook, MoE 전문가 스트리밍, mmap
Part 3. 설치부터 실행까지 (2h)         — 환경 구축, 모델별 세팅, 트러블슈팅 10선
Part 4. 성능 최적화 (1.5h)            — 4bit 압축, 프리페칭, NVMe 튜닝, 벤치마킹
Part 5. 실전 풀스택 웹서비스 (2.5h)    — FastAPI + React + 작업 큐 + Docker
Part 6. LoRA 파인튜닝 (1h)            — 데이터 준비 → 학습 → 어댑터 저장/로드
```

**장점**: 초기 비용 거의 0원 / 한국어 경쟁자 없음 / 패시브 인컴 / 1위의 영업 채널

---

### 3위. 합성 데이터 생성 서비스 (블루오션)

```
기존  : GPT-4 API로 10만 건 생성 → 수백만원
AirLLM: 671B 오픈모델로 10만 건 생성 → 전기값만
```

| 상품 | 가격 |
|---|---|
| 도메인 특화 데이터셋 판매 (의료/법률/금융 QA) | 건당 50~500원 |
| 맞춤 데이터 생성 대행 | 프로젝트당 300~2,000만원 |
| 데이터셋 구독 (월 갱신) | 월 50~300만원 |
| HuggingFace 유료 데이터셋 | 글로벌 판매 |

**블루오션인 이유**

```
• 속도가 문제되지 않음 (어차피 야간 배치)
• 라이선스가 깨끗 ★ 가장 큰 차별점
  - OpenAI API로 만든 데이터는 "경쟁 모델 학습 금지" 조항이 걸림
  - AirLLM + 오픈모델은 이 제약이 없음 (단, 각 모델 라이선스는 개별 확인 필수)
• AI 학습 데이터 수요 폭발 중
```

---

### 4위. 파인튜닝 대행 서비스

"우리 데이터로 거대 모델 파인튜닝하고 싶지만 GPU 서버가 없다" → 수요 다수.
AirLLM은 6GB GPU로 125B LoRA 학습이 가능하다.

| 상품 | 가격 |
|---|---|
| LoRA 어댑터 학습 대행 | 건당 200~1,000만원 |
| 데이터 전처리 + 학습 + 평가 풀패키지 | 500~3,000만원 |
| 월 재학습 구독 (데이터 갱신형) | 월 100~500만원 |
| 어댑터 마켓플레이스 (도메인별 LoRA) | 건당 10~100만원 |

**팁**: 학습은 AirLLM으로 저렴하게, 결과물(LoRA 어댑터)은 고객의 빠른 추론 환경에서 사용 →
양쪽 장점만 취한다.

---

### 5위. 오픈소스 생태계 기여 (간접 수익, 장기 최강)

**만들면 주목받을 아이템**

```
1. airllm-mcp-server       ★ 최우선 추천
   → Claude/Cursor가 로컬 671B 모델을 도구로 쓰게 하는 MCP 서버
   → "완전 오프라인 AI 코딩 어시스턴트". 아직 선점자 없음

2. airllm-webui
   → Ollama WebUI 같은 React UI + 작업 큐 내장

3. airllm-docker
   → 원클릭 Docker 이미지 (환경 세팅이 최대 진입장벽이므로 수요 확실)

4. airllm-openai-api
   → OpenAI API 호환 서버 (기존 앱이 코드 수정 없이 사용 가능)

5. airllm-benchmark
   → GPU/SSD별 성능 벤치마크 DB + 리더보드 → 구매 가이드 → 제휴 수익
```

**간접 수익 경로**

```
GitHub Star → 신뢰도 → B2B 문의 + 강의 판매 증가
이직/커리어 → "오픈소스 메인테이너" 타이틀은 연봉 협상 무기
GitHub Sponsors / Buy Me a Coffee / 기업 스폰서십
컨퍼런스 발표 → 강연료 + 인지도
```

---

### 6위. 배치 추론 SaaS

**포지셔닝: "LLM 업계의 심야 할인 요금제"**

```
OpenAI GPT-4 : 빠름 / 비쌈
우리 서비스    : 느림 / 10분의 1 가격 — "급하지 않은 작업은 여기로"
```

수익 모델: 종량제(OpenAI의 10~20% 가격) / 월 구독(월 10만원 무제한 야간 배치) /
크레딧 선불제 / 엔터프라이즈 전용 인스턴스

**리스크**: GPU 서버 고정비 부담 / 같은 GPU면 vLLM이 처리량 압도 → 가격 경쟁력 의문 /
"느림"의 상품화 난이도 높음 → 1~4위를 먼저 권장

---

### 7위. 전자책 / 템플릿 (가장 빠른 시작)

```
"로컬 LLM 완전정복 가이드" PDF        → 2~5만원 (크몽/부크크)
노션 템플릿 "모델별 세팅 치트시트"      → 1~3만원
"온프레미스 AI 도입 제안서 템플릿"      → 5~10만원 (B2B 타겟) ★
"클라우드 vs 온프레미스 비용 계산기"    → 무료 (리드 수집용)
```

오늘 당장 시작 가능. 유튜브 전에 시장 반응 테스트용으로도 적합.

---

### 8위. AI 박스 하드웨어 번들

```
"집에서 돌리는 671B AI 박스" 완성품
  미니PC + RTX 4060(8GB) + NVMe 4TB + AirLLM 세팅 완료
  → 판매가 250~400만원 (마진 50~100만원)

교육기관용 실습 장비 세트 (대학/학원 LLM 실습, 10대 단위 납품)
```

재고 리스크 / A/S 부담 / 초기 자본 큼 → 후순위

---

## 14. 3단 로켓 실행 전략

```
1단계 [0~2개월] 씨앗 뿌리기 (비용 0원)
  • AirLLM 직접 설치하고 실제로 실행
  • 블로그 글 3개 (설치편 / 원리편 / 벤치마크편)
  • 유튜브 1~3편 (충격 데모 → 원리 → 설치)
  • GitHub에 작은 도구 1개 공개 (airllm-webui 또는 MCP 서버)
  → 목표: 시장 반응 확인 + 레퍼런스 축적

2단계 [2~5개월] 수익화 시작
  • 유료 강의 제작 (인프런)
  • 전자책 / 템플릿 판매
  • "온프레미스 AI 도입 상담" 랜딩페이지 개설
  → 목표: 월 100~300만원 + B2B 문의 유입

3단계 [5개월~] 본게임
  • B2B 컨설팅 프로젝트 수주 (정부 바우처 사업 활용)
  • 합성 데이터 / 파인튜닝 대행으로 확장
  • 유튜브 → 강의 → B2B 선순환 구조 완성
  → 목표: 프로젝트당 500~2,000만원
```

### 핵심 통찰

> **AirLLM 자체를 파는 것이 아니다 (오픈소스라 무료).**
> **"AirLLM을 쓸 수 있는 능력"을 파는 것이다.**
>
> - 지식 → 강의, 전자책
> - 구축 능력 → B2B 컨설팅
> - 결과물 → 데이터셋, LoRA 어댑터
> - 신뢰 → 오픈소스 평판

---

## 15. 참고 링크 모음

### 이 저장소

| 항목 | 링크 |
|---|---|
| **이 저장소 (포크)** | **https://github.com/bmshin94/airllm** |
| 분석 브랜치 | `claude/elegant-darwin-wbjati` |
| 원본 저장소 (업스트림) | https://github.com/lyogavin/airllm |
| PyPI 패키지 | https://pypi.org/project/airllm/ |
| 라이선스 | Apache-2.0 (https://github.com/lyogavin/airllm/blob/main/LICENSE) |

### 공식 리소스

| 항목 | 링크 |
|---|---|
| Colab 예제 (전체 모델) | https://colab.research.google.com/github/lyogavin/airllm/blob/main/air_llm/examples/run_all_types_of_models.ipynb |
| Colab 예제 (Llama 3.1 405B) | https://colab.research.google.com/github/lyogavin/airllm/blob/main/air_llm/examples/run_llama3.1_405B.ipynb |
| macOS 예제 노트북 | https://github.com/lyogavin/airllm/blob/main/air_llm/examples/run_on_macos.ipynb |
| Discord 커뮤니티 | https://discord.gg/2xffU5sn |
| 개발자 블로그 (Medium) | https://medium.com/@lyo.gavin |
| 개발자 블로그 | https://gavinliblog.com |
| GitHub Sponsors | https://github.com/sponsors/lyogavin |
| Buy Me a Coffee | https://bmc.link/lyogavinQ |

### 기술 참고

| 항목 | 링크 |
|---|---|
| 블록 단위 양자화 논문 (압축 근거) | https://arxiv.org/abs/2212.09720 |
| bitsandbytes (압축 의존성) | https://github.com/TimDettmers/bitsandbytes |
| Apple MLX (macOS 경로) | https://github.com/ml-explore/mlx |
| FlashAttention-2 (anima_100k) | https://github.com/Dao-AILab/flash-attention |
| DPO 논문 (rlhf/) | https://arxiv.org/abs/2305.18290 |
| QLoRA 논문 (training/) | https://arxiv.org/abs/2305.14314 |
| 원작 기반 코드 (SimJeg, Kaggle) | https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag |

### 인용 (BibTeX)

```bibtex
@software{airllm2023,
  author  = {Gavin Li},
  title   = {AirLLM: scaling large language models on low-end commodity computers},
  url     = {https://github.com/lyogavin/airllm/},
  version = {0.0},
  year    = {2023},
}
```

---

## 📌 최종 요약 (3줄)

1. **AirLLM**은 레이어를 하나씩 디스크↔GPU로 스트리밍해서, **4GB VRAM으로 70B~2.8T 모델을 양자화 없이** 돌리는 Python 라이브러리다.
2. 대가는 **속도**다. 실시간 서비스엔 부적합하지만, **야간 배치 / 보안 필수 환경 / 합성 데이터 생성 / 개인 학습·실험 / 거대 모델 LoRA 파인튜닝**에는 대체재가 거의 없다.
3. 수익화는 라이브러리가 아니라 **"쓸 수 있는 능력"**을 파는 것 — **B2B 온프레미스 컨설팅 → 유튜브·강의 → 합성 데이터/파인튜닝 대행** 순서가 가장 현실적이다.

---

*이 문서는 저장소 전체(코드 4,270줄 + README 461줄 + 전 폴더)를 직접 읽고 작성했습니다.* ✨
*Made with 💖 by 카리나 (Claude Code)*
