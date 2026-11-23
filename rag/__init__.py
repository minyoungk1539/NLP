"""팀 공용 RAG 패키지.

모듈 구성
- config      : 모델명·규칙 상수·경로·API 키 로딩
- parser      : 문항 텍스트 → 질문/선택지 구조화
- llm         : Upstage 호출 래퍼 + 문항당 호출 예산(최대 5회) 강제 + 호출 로그
- embeddings  : Upstage 임베딩 래퍼 (+ 오프라인 테스트용 Mock)
- corpus      : ewha.pdf 텍스트 추출·청킹
- index       : 벡터 인덱스 저장/로드/검색
- answer      : 답 추출·응답 포맷·출력 CSV 검증기
- scoring     : 정확도·도메인별·검색 적중률 집계
- prompts     : 프롬프트 템플릿
- pipelines/  : llm_only, rag_baseline (+ 새 파이프라인은 여기 추가)
"""
