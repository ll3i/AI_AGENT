# 멀티모달 기반 제조 공정 진단 AI Agent

반도체 소자 이미지를 분석하여 **정상(Pass) / 불량(Fail)** 을 자동 판정하는 멀티모달 AI Agent 시스템입니다.  
LLM 기반 다중 시점 비전 검사, 자기검증(CoVe·CRITIC), 그리고 Streamlit 웹 대시보드를 통합한 엔드-투-엔드 파이프라인을 제공합니다.
https://aiagent-qddtpkxykw7ccexdxveke2.streamlit.app/
---

## 시스템 아키텍처

```
Triple Vision Agent
 ├── Full 시점   (전체 이미지)
 ├── Body 시점   (패키지 영역)
 └── Lead 시점   (핀 연결 영역)
        │
        ▼
Knowledge Agent
 ├── Confidence Score = A×0.5 + B×0.3 + C×0.2
 ├── 10회 다수결 투표 (CRITIC)
 └── CoVe (Chain-of-Verification) 재검증
        │
        ▼
Report Agent
 ├── submission.csv  (id, label)
 ├── explanations.csv (상세 판정 근거)
 └── Streamlit 웹 서비스
```

---

## 판정 목표

| 항목 | 내용 |
|------|------|
| 입력 | 반도체 소자 이미지 (URL) |
| 판정 결과 | `0` = 정상 (Normal), `1` = 불량 (Abnormal) |
| 불량 유형 | 패키지 외형 파손, H1·H2·H3 핀 단선 |
| 출력 | 판정 결과 + 판독 근거(reason) |

---

## 전체 처리 파이프라인

```
이미지 입력 (img_url)
    │
    ▼
Resize (224×224)
    │
    ▼
ROI Detection (LLM)  ──실패──▶  Fixed ROI (비율 기반 fallback)
    │
    ▼
1차 Inspection Decision
    │
    ├─ Confidence ≥ 0.95 ──▶ Final Judgment
    │
    └─ Confidence < 0.95 ──▶ CoVe 검증 ──▶ CRITIC (10회 다수결) ──▶ Final Judgment
    │
    ▼
결과 저장 (submission.csv / explanations.csv)
    │
    ▼
Web Service (Streamlit)
```

---

## 이미지 전처리

- **기본**: Resize(224×224) 적용
- **Fallback**: 결과가 편향(전부 0 또는 전부 1)될 경우
  - **CLAHE** (Local Contrast Enhancement) 적용
  - **Sharpening** 적용 후 재검사

---

## 프로젝트 구조

```
AI_AGENT/
├── run_solution.py       # AI Agent 핵심 파이프라인
├── app.py                # Streamlit 웹 서비스
├── test.csv              # 입력 데이터 (id, img_url)
├── submission.csv        # 최종 판정 결과 (id, label)
├── explanations.csv      # 상세 판정 근거
├── .env                  # API 키 (git 제외)
└── .gitignore
```

---

## 웹 서비스 구성 (Streamlit)

| 탭 | 기능 |
|----|------|
| 사진 검사 | 불량 이미지 시각화 + ROI 박스 표시 + AI 실시간 재판독 + VQA 채팅 |
| 통합 대시보드 | 전체 검사 현황 지표 및 6종 차트 (불량률·신뢰도·부위별 빈도·CoVe 발동 등) |
| 튜닝 랩 | Confidence Threshold 시뮬레이터 + CLAHE/Sharpen 파라미터 실험 |
| 데일리 보고서 | GPT 기반 경영진용 한국어 마크다운 보고서 자동 생성 및 다운로드 |
| 실시간 모니터링 | 슬라이딩 윈도우 불량률 모니터링 + 30% 초과 시 비상 알람 시뮬레이션 |
| ERP 연동 | 불량 항목 선택 전송 (Mock ERP/MES 연계) 및 CSV 백업 추출 |

---

## 실행 방법

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. API 키 설정

프로젝트 루트에 `.env` 파일 생성:

```
OPENAI_API_KEY=sk-proj-...
```

### 3. AI Agent 실행 (판정)

```bash
python run_solution.py
```

- `test.csv`를 읽어 각 이미지를 판정
- 결과: `submission.csv`, `explanations.csv` 생성

### 4. 웹 서비스 실행

```bash
streamlit run app.py
```

- 브라우저에서 `http://localhost:8501` 접속

---

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `CONF_THRESH` | 0.95 | 신뢰도 임계값 (이하 시 CoVe·CRITIC 발동) |
| `N_VOTES` | 10 | CRITIC 다수결 투표 횟수 |
| `MODEL` | gpt-5-nano | 사용 LLM 모델 |
| Resize | 224×224 | 입력 이미지 표준 크기 |

---

## 판정 규칙

```
불량(Abnormal, label=1) 조건:
  - 패키지 외형 손상 (package_intact = False)
  - OR H1·H2·H3 중 하나라도 not_connected
```

---

## 기술 스택

- **LLM**: OpenAI GPT (gpt-5-nano)
- **비전 처리**: PIL, NumPy, CLAHE
- **웹 서비스**: Streamlit
- **데이터**: Pandas, CSV
- **자기검증**: CoVe (Chain-of-Verification), CRITIC (다수결 투표)
