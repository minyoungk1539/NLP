"""위키백과 검색 (과제 규칙: https://en.wikipedia.org/w/api.php 만 사용).

- list=search로 문서 제목을 찾고, prop=extracts(explaintext)로 본문 앞부분을 가져온다.
- 규칙상 '청킹·임베딩 전 원본'을 제출해야 하므로, 모든 API 응답을 요청 URL·수집 시각과 함께
  data/wiki_raw/<sha1>.json 에 그대로 저장한다. 같은 요청은 이 캐시를 다시 쓴다 (재현성).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

from . import config

API = "https://en.wikipedia.org/w/api.php"
RAW_DIR = config.DATA_DIR / "wiki_raw"
USER_AGENT = "NLP-course-RAG-project/0.1 (educational use)"


class WikiClient:
    def __init__(self, raw_dir: Path = RAW_DIR, extract_chars: int = 1500, min_interval: float = 1.0):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.extract_chars = extract_chars
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def _get(self, params: dict) -> dict:
        params = {**params, "format": "json", "formatversion": "2"}
        url = API + "?" + urllib.parse.urlencode(sorted(params.items()))
        path = self.raw_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        # 위키백과는 동시 요청에 429를 준다: 모든 요청을 한 줄로 세우고(lock) 간격을 둔다.
        # 429·5xx면 Retry-After(없으면 지수 백오프)만큼 기다렸다 다시 보낸다.
        with self._lock:
            for attempt in range(6):
                wait = self.min_interval - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        data = json.loads(r.read().decode("utf-8"))
                    break
                except urllib.error.HTTPError as e:
                    if attempt == 5 or e.code not in (429, 500, 502, 503, 504):
                        raise
                    retry_after = e.headers.get("Retry-After")
                    time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else 5 * 2 ** attempt)
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 5:
                        raise
                    time.sleep(2 ** attempt)
        path.write_text(json.dumps({"url": url, "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                    "response": data}, ensure_ascii=False), encoding="utf-8")
        return data

    def search(self, query: str, limit: int = 3) -> list[str]:
        data = self._get({"action": "query", "list": "search", "srsearch": query[:300], "srlimit": limit})
        return [h["title"] for h in data.get("query", {}).get("search", [])]

    def extracts(self, titles: list[str]) -> dict[str, str]:
        if not titles:
            return {}
        data = self._get({"action": "query", "prop": "extracts", "explaintext": "1", "exintro": "1",
                          "redirects": "1", "titles": "|".join(titles)})
        return {p["title"]: (p.get("extract") or "")[: self.extract_chars]
                for p in data.get("query", {}).get("pages", []) if p.get("extract")}

    def passages(self, queries: list[str], n_pages: int = 3, max_total: int = 5) -> list[dict]:
        titles: list[str] = []
        for q in queries:
            for t in self.search(q, limit=n_pages):
                if t not in titles:
                    titles.append(t)
        titles = titles[:max_total]
        ex = self.extracts(titles)
        return [{"title": t, "text": ex[t]} for t in titles if t in ex]
