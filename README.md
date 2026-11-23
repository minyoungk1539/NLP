# RAG 객관식 QA: 팀 공용 코드

모든 팀원이 이 뼈대를 공유한다. 각자 바꾸는 것은 `.env`의 API 키뿐이다.
LLM only와 Baseline RAG는 이미 구현되어 있다. 개선 실험은 `rag/pipelines/`에 새 파일로 추가한다.

## 1. 빠른 시작

모든 명령은 `Project/` 폴더에서 실행한다.

```bash
pip install -r requirements.txt
cp .env.example .env            # 그리고 .env에 UPSTAGE_API_KEY=... 입력

python tests/smoke_test.py      # API 없이 기본 점검 (파서·답 추출·호출 한도·형식 검증)
python run.py --mock --input data/dev/dev_all.csv --output out.csv   # API 없이 전체 흐름 시험

python scripts/build_index.py   # 학칙 인덱스 생성 (한 명만 하고 index/ 폴더를 공유)
python run.py --pipeline llm_only     --input data/dev/dev_all.csv --output out_llm.csv
python run.py --pipeline rag_baseline --input data/dev/dev_all.csv --output out_rag.csv
```

공식 실행·채점 명령은 아래와 같다.

```bash
python run.py --input data/test_samples.csv --output responses.csv
python eval.py responses.csv data/test_samples.csv
```

> ⚠ **처음 실제 API로 돌리는 사람이 확인할 것**: `python run.py --pipeline llm_only --limit 3`
> - `reasoning_effort="none"`을 모델이 거부하면 자동으로 빼고 다시 보낸다. 콘솔에 안내가 뜨면 팀에 공유한다.
> - 임베딩 모델 이름(`solar-embedding-2-passage`, `solar-embedding-2-query`)이 통하는지 확인한다.
> - 429 오류가 잦으면 `--workers 2`로 줄인다.

## 2. 폴더 구조

```
Project/
├── run.py                 공식 진입점 (--input/--output, 입력 순서 유지, index=False 저장)
├── eval.py                공식 채점기 (수정 금지)
├── rag/
│   ├── config.py          모델명·호출 한도·max_tokens 상한·경로 (규칙 값은 여기에만 있음)
│   ├── parser.py          문항 파서: 본문/선택지 분리, 선택지 개수, 도메인 추정, ID 부여
│   ├── llm.py             호출 래퍼: 문항당 5회 한도 강제, 모델 고정, 호출 로그
│   ├── embeddings.py      Upstage 임베딩 (passage/query 구분) + 오프라인 Mock
│   ├── corpus.py          ewha.pdf 추출, 쪽 머리말 제거, 고정 길이 청킹
│   ├── index.py           벡터 인덱스 저장/로드/검색 (numpy)
│   ├── answer.py          답 추출, "[ANSWER]: (X)" 포맷, 출력 CSV 검증기
│   ├── prompts.py         프롬프트 (LLM only와 RAG는 컨텍스트 유무만 다름)
│   ├── scoring.py         정확도(전체·도메인·유형), 호출 수, 검색 근거쪽 적중률
│   └── pipelines/         llm_only.py · rag_baseline.py · _template.py(복사용)
├── scripts/
│   ├── build_index.py     인덱스 생성 (--size/--overlap/--name/--dump)
│   ├── check_output.py    제출 CSV 형식 검증 + 정확도
│   ├── search_index.py    검색만 따로 확인 (LLM 호출 없음)
│   ├── inspect_run.py     오답·검색 결과 분석, 두 실행 비교
│   └── make_dev.py        MMLU-Pro 개발 표본 다운로드 + dev_all.csv 합치기
├── tests/smoke_test.py    오프라인 점검
├── data/
│   ├── test_samples.csv, test_samples_students.csv, ewha.pdf   (받은 자료)
│   └── dev/               개발셋 (아래 4절)
├── index/                 인덱스 (실제 임베딩 인덱스는 제출 ZIP에 포함)
└── runs/                  실행 로그 (git 제외)
```

## 3. 규칙이 코드에서 어떻게 지켜지나

| 규칙 | 코드에서 지키는 방식 |
|---|---|
| `solar-mini4-260922`만 사용 | `config.CHAT_MODEL` 고정. `llm.chat()`에 `model`을 넘기면 에러 |
| 문항당 LLM 호출 ≤ 5회 | `QuestionSession`이 횟수를 센다. 6번째 호출은 `BudgetExceeded` 발생. API 오류로 실패한 호출도 1회로 센다 |
| `max_tokens` ≤ 4096 | 넘기면 에러. 한 번에 여러 답을 받는 `n>1`도 금지 |
| 임베딩·검색은 호출 수에서 제외 | 임베딩은 `embeddings.py`를 거치므로 `llm.chat` 집계에 안 들어간다 |
| 출력 `question_id,responses`, 태그 정확히 1개 | 응답을 `[ANSWER]: (X)` 한 줄로만 저장. 설명은 `runs/` 로그에 남긴다. 저장 직후 검증기가 자동으로 돈다 |
| 입력 순서 유지, 빠진 행 없음 | 병렬 처리해도 입력 위치에 기록한다. 호출·추출이 실패해도 fallback 답을 넣고 `fallback=True`로 표시한다 |
| 답은 그 문항의 선택지 범위 안 | 파서가 문항마다 선택지 글자를 계산한다 (4·7·10지선다) |
| API 키는 제출물에 넣지 않음 | `.env`에서만 읽는다. `.env`는 `.gitignore`에 있다 |

> 파이프라인 안에서 OpenAI 클라이언트를 직접 부르면 호출 수가 집계되지 않는다. **반드시 `llm.chat()`으로만 호출한다.**

## 4. 개발셋 (`data/dev/`)

| 파일 | 문항 | 내용 |
|---|---|---|
| `dev_official.csv` | 25 | 공식 연습 문항에 근거 조문(`gold_ref`), 근거쪽(`gold_page`), 유형(`qtype`) 주석을 단 것 (`S-001`~) |
| `dev_ewha.csv` | 35 | 학칙에서 직접 만든 4지선다 (`E-001`~). 정답은 ewha.pdf 원문으로 확인했다. 유형: 단순 사실, 조건 적용, 여러 조문 비교, 계산, 부칙, 개정 날짜, '아닌 것' 고르기, 영어 질문 2개 |
| `dev_mmlu.csv` | 75 | MMLU-Pro test에서 law/psychology/business/philosophy/history 과목별 15개씩 뽑음 (seed 42). 연습 문항과 같은 문제는 뺐다 |
| `dev_all.csv` | 135 | 위 셋을 합친 것 (이화 47, MMLU 88). **기본 비교는 이 파일로 한다** |

- `qtype` 값: `fact`, `condition`, `multi`, `calc`, `supplementary`(부칙), `amendment`(개정 날짜), `negation`, `no_evidence`(MMLU 문항은 `knowledge`, `passage`, `application`)
- `gold_page`가 있는 문항은 검색 결과에 근거쪽이 들어왔는지(**근거쪽 적중률**)가 자동으로 집계된다
- 🚫 **개발셋을 검색 소스(인덱스)에 넣으면 규칙 위반(0점)이다.** 평가용으로만 쓴다
- 연습 영어 문항 18개 중 15개가 MMLU-Pro test에 그대로 있었다. 숨겨진 평가셋도 같은 출처일 가능성이 높다
- 공식 `S-004`(복수전공 신청 자격)는 학칙 본문과 부칙에서 근거를 찾지 못했다
- 예전 샘플 `test_samples_students.csv` 1번(영어·정보 인증 학점)의 정답 키는 (A) 인정 안됨이다. 그런데 학칙 제48조의2①은 "각 3학점"이라 (D)가 맞다. 예전 이화 2–5번(LMS 등)은 학칙에 없는 내용이라 개발셋에 넣지 않았다

## 5. 새 방법 실험하기

1. `rag/pipelines/_template.py`를 `rag/pipelines/<이름>.py`로 복사한다. 클래스 이름과 `name`을 바꾼다
2. `rag/pipelines/__init__.py`의 `PIPELINES`에 등록한다
3. `python run.py --pipeline <이름> --input data/dev/dev_all.csv --output out.csv --run-name "메모"`

- 검색을 바꾸려면(청크 크기, 조 단위 청킹, BM25 등) 새 인덱스를 다른 `--name`으로 만들고 `--index`로 고른다
- 검색 결과는 `pred.info["retrieved"] = [hit.brief() ...]`에 넣는다. 그래야 근거쪽 적중률이 집계된다
- 비교는 항상 **같은 입력 파일**로 한다. Mock 결과(`mock=True`)는 성능 비교에 쓰지 않는다

## 6. 실행 기록과 실패 분석

실행할 때마다 `runs/<시각>_<파이프라인>[_메모]/`에 다음 파일이 생긴다.

| 파일 | 내용 |
|---|---|
| `config.json` | 인자, 생성 파라미터, 인덱스 정보 |
| `results.jsonl` | 문항별 예측, 정답, 호출 수, 검색된 쪽, 근거쪽 적중, 모델 원출력 |
| `calls.jsonl` | 호출별 프롬프트, 출력, 토큰 사용량, 지연시간 |
| `summary.json` | 정확도 등 요약 |

`runs/leaderboard.csv`에는 실행마다 한 줄씩 쌓인다.

```bash
python scripts/inspect_run.py                         # 최근 실행의 오답 + 검색된 쪽 + 근거쪽 적중 여부
python scripts/inspect_run.py runs/<폴더> --domain ewha --raw
python scripts/inspect_run.py --compare runs/<A> runs/<B>   # 두 방법에서 결과가 갈린 문항
python scripts/search_index.py --qid E-015 --input data/dev/dev_ewha.csv   # 이 문항의 검색 결과만
```

`inspect_run.py` 결과로 실패 유형을 가른다 (발표 슬라이드의 실패 유형 4가지).
- 근거쪽 미검색 → Missing source / Wrong passage
- 근거쪽은 검색됐는데 오답 → Missing condition / Incorrect application

## 7. Baseline에서 일부러 하지 않은 것 (개선 실험 몫)

- **청킹**: 줄 이어 붙이기, 조(條) 단위 청킹, 개정 날짜 메타데이터 없음. 고정 800자, 겹침 150자만 쓴다
  - 예: E-001의 근거 문장 "학기개시일로부터 90일 이내에 휴학을 신청하여야 한다"는 청크 경계에서 잘린다
- **표**: p.9와 p.20의 성적점 표는 열 단위로 추출돼 깨진다 (S-003 관련)
- **검색**: 도메인 라우팅 없이 MMLU 문항에도 학칙 검색 결과를 붙인다. BM25·하이브리드·재순위화 없음
- **생성**: LLM 1회 호출, 자기검증·다수결 없음

## 8. 제출 전 체크리스트

- [ ] `run.py`의 `DEFAULT_PIPELINE`을 최종 방법으로 바꾼다
- [ ] `python run.py --input data/test_samples.csv --output responses.csv` 실행 후 `python eval.py responses.csv data/test_samples.csv`
- [ ] `python scripts/check_output.py responses.csv --input data/test_samples.csv`가 통과한다
- [ ] 실제 임베딩 인덱스 폴더(`index/<이름>/`)가 ZIP에 들어 있다. `mock_` 인덱스는 넣지 않는다
- [ ] `.env`, `runs/`, `data/dev/_cache/`는 ZIP에서 뺀다
- [ ] 새 가상환경에서 `pip install -r requirements.txt`부터 다시 돌려본다. requirements는 `==`로 고정한다
- [ ] 외부 자료를 썼다면 원본을 클라우드에 올리고 README에 링크, 출처 URL, 수집 날짜를 적는다
