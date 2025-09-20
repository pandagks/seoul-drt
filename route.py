# ==== route.py (추가) ==================================================
import math
import pandas as pd
from typing import Dict, Tuple, Optional, List
import heapq
import re

def normalize_stop_id(s: str) -> str:
    if s is None:
        return s
    s = str(s).replace("\u00A0", " ").strip()          # non‑breaking space 정리
    # "21_ 극동아파트" → "21_극동아파트" (숫자+언더스코어 다음의 공백 제거)
    s = re.sub(r"^(\d+_)\s+", r"\1", s)
    # 중복 공백 압축
    s = re.sub(r"\s{2,}", " ", s)
    return s

# 전역 거리 테이블: (a,b) -> km
_DIST_KM: Dict[Tuple[str, str], float] = {}
_NODES: List[str] = []

def _key(a: str, b: str) -> Tuple[str, str]:
    return (a, b)

def load_distance_matrix_csv(path: str, unit: str = "m") -> None:
    """
    distance_matrix.csv (index=정류장ID, columns=정류장ID, 값=거리) 로딩
    unit: 'm' or 'km'  (CSV가 미터면 내부에서 km로 변환)
    """
    global _DIST_KM, _NODES
    df = pd.read_csv(path, index_col=0)
    _NODES = df.index.astype(str).tolist()
    to_km = (1.0/1000.0) if unit == "m" else 1.0

    _DIST_KM.clear()
    for a in df.index:
        row = df.loc[a]
        for b, v in row.items():
            if pd.notna(v):
                _DIST_KM[_key(str(a), str(b))] = float(v) * to_km

def get_distance_between(a: str, b: str):
    a = normalize_stop_id(a)
    b = normalize_stop_id(b)
    if a == b:
        return 0.0
    return _DIST_KM.get(_key(a, b))

def _neighbors(u: str):
    """그래프 이웃 iter (v, dist_km)"""
    for v in _NODES:
        d = _DIST_KM.get(_key(u, v))
        if d is not None and math.isfinite(d) and d > 0:
            yield v, d

def shortest_path_distance_km(a: str, b: str) -> Optional[float]:
    """
    Dijkstra로 최단거리(km). 경로 없으면 None.
    """
    if a not in _NODES or b not in _NODES:
        return None
    if a == b:
        return 0.0

    dist = {n: math.inf for n in _NODES}
    dist[a] = 0.0
    pq = [(0.0, a)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == b:
            return d
        for v, w in _neighbors(u):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return None

# route.py

import math
import pandas as pd
from typing import Dict, Tuple, Optional, List
import heapq

# ... (기존 _DIST_KM, _NODES, _key, load_distance_matrix_csv, get_distance_between, shortest_path_distance_km 그대로)

def _fallback_adjacent_distance_km(stop_order: List[str]) -> float:
    """
    인접 간선이 없을 때 사용할 대체 거리(km)를 계산.
    1순위: stop_order 상에서 이미 '알려진 인접 거리'들의 중앙값
    2순위: 매트릭스 내 모든 유한 거리의 중앙값
    둘 다 없으면 명확히 에러를 내서 매트릭스를 먼저 보강하도록 유도.
    """
    known_adj = []
    for i in range(len(stop_order) - 1):
        a, b = stop_order[i], stop_order[i + 1]
        d = _DIST_KM.get((a, b))
        if d is None:
            d = _DIST_KM.get((b, a))
        if d is not None and math.isfinite(d) and d > 0:
            known_adj.append(float(d))

    pool = known_adj
    if not pool:
        # 전역 유한 거리에서 중앙값
        pool = [float(v) for v in _DIST_KM.values() if math.isfinite(v) and v > 0]

    if not pool:
        raise ValueError("[route] 유효한 거리 데이터가 없습니다. load_distance_matrix_csv(...)로 먼저 로드하세요.")

    pool.sort()
    return pool[len(pool) // 2]  # 중앙값

def ensure_consecutive_edges(stop_order: List[str]) -> None:
    """
    stop_order의 인접쌍 (i, i+1)에 직접거리가 없으면:
      1) 그래프 최단경로로 보간하여 채움
      2) 그래도 없으면, 인접 간선 중앙값(또는 전역 중앙값)으로 폴백하여 채움
    """
    # 미리 한 번만 계산해두고 재사용
    fallback_val: Optional[float] = None

    for i in range(len(stop_order) - 1):
        a, b = stop_order[i], stop_order[i + 1]
        if get_distance_between(a, b) is not None:
            continue

        # 1) 최단경로로 보간 시도
        d = shortest_path_distance_km(a, b)
        if d is not None and math.isfinite(d) and d > 0:
            _DIST_KM[(a, b)] = d
            _DIST_KM[(b, a)] = d
            continue

        # 2) 폴백(중앙값) 사용
        if fallback_val is None:
            fallback_val = _fallback_adjacent_distance_km(stop_order)
        _DIST_KM[(a, b)] = fallback_val
        _DIST_KM[(b, a)] = fallback_val
        print(f"[route][WARN] 인접 거리 미존재: {a}→{b} → 중앙값 {fallback_val:.3f} km로 대체")

