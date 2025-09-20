# ga_optimizer.py
# "픽업 고정 순서 + 중간 드롭 삽입" 규칙 기반 경로 생성기
import math
from typing import List, Tuple, Dict, Set
from route import get_distance_between
from parameters import DEPOT_ID

SPEED_MIN_PER_KM = 3  # 1km → 3분

def _order_key(stop_id: str) -> int:
    # "12_상도역..." → 12 를 반환 (파싱 실패 시 큰 값)
    try:
        return int(str(stop_id).split("_", 1)[0])
    except Exception:
        return 10**9

def _dist(a: str, b: str) -> float:
    d = get_distance_between(a, b)
    return float(d) if (d is not None and math.isfinite(d)) else float("inf")

def _unique(seq: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def _build_rule_based_sequence(pairs: List[Tuple[str, str]]) -> List[str]:
    """
    규칙:
    1) 출발은 DEPOT_ID(00)에서
    2) 픽업 정류장들은 정류장아이디 숫자 기준 오름차순(고정 순서)
    3) 각 픽업 사이에, '이미 픽업된 승객들의 하차' 중
       현재 위치→하차 < 현재 위치→다음 픽업 인 경우, 그 하차를 먼저 방문 (가까운 순으로 반복)
    4) 모든 픽업 끝나면, 남은 하차는 현재 위치에서 가까운 순으로 처리
    5) 동일 정류장은 한 번만 방문
    """
    if not pairs:
        return [DEPOT_ID]

    # 1) 픽업/하차 목록 정리
    pickups_ordered = sorted({p for p, _ in pairs}, key=_order_key)

    # 각 픽업이 만들어내는 '하차 후보'를 모아두기
    dropoffs_by_pickup: Dict[str, List[str]] = {}
    for p, d in pairs:
        dropoffs_by_pickup.setdefault(p, []).append(d)

    seq: List[str] = [DEPOT_ID]
    current = DEPOT_ID
    eligible_dropoffs: List[str] = []  # '이미 픽업한 승객'들의 하차 후보 모음(중복 허용)

    for idx, next_pick in enumerate(pickups_ordered):
        # 2) 다음 픽업 가기 전, 가까운 하차 먼저 들르기 (반복)
        while True:
            if not eligible_dropoffs:
                break
            # 현 위치에서 가장 가까운 하차 찾기
            best_d = min(eligible_dropoffs, key=lambda x: _dist(current, x))
            if _dist(current, best_d) < _dist(current, next_pick):
                seq.append(best_d)
                current = best_d
                # 해당 하차를 목록에서 모두 제거(그 정류장에 한 번만 들르면 충분)
                eligible_dropoffs = [x for x in eligible_dropoffs if x != best_d]
            else:
                break

        # 3) 이제 다음 픽업으로 이동
        seq.append(next_pick)
        current = next_pick
        # 방금 픽업에서 생긴 하차 후보 추가
        eligible_dropoffs += dropoffs_by_pickup.get(next_pick, [])

    # 4) 마지막 픽업 후, 남은 하차를 가까운 순으로 모두 처리
    while eligible_dropoffs:
        best_d = min(eligible_dropoffs, key=lambda x: _dist(current, x))
        seq.append(best_d)
        current = best_d
        eligible_dropoffs = [x for x in eligible_dropoffs if x != best_d]

    # 5) 동일 정류장 중복 제거(첫 등장은 유지)
    seq = _unique(seq)
    return seq

def _evaluate_path_distance_km(path: List[str]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        d = get_distance_between(path[i], path[i+1]) or 0.0
        total += d
    return total

def run_ga(pairs: List[Tuple[str, str]], generations=0, pop_size=0, verbose=True, plot=False):
    """
    기존 GA 대신, 규칙 기반 경로를 생성해서 동일한 반환 포맷을 유지.
    반환: (stops_to_visit, fitness_with_return, total_distance_km, total_minutes)
    """
    path = _build_rule_based_sequence(pairs)

    # 이동거리/시간 계산
    total_distance = _evaluate_path_distance_km(path)
    total_minutes = int(round(total_distance * SPEED_MIN_PER_KM))

    # 'fitness_with_return' 인터페이스 유지(단순히 경로 1개만 평가치로)
    fitness_with_return = [total_distance]

    if verbose:
        print("[RULE] 경로:", " → ".join(path))
        print(f"[RULE] 총 이동 거리: {total_distance:.2f} km")
        print(f"[RULE] 총 예상 소요 시간: {total_minutes}분")

    return path, fitness_with_return, total_distance, total_minutes
