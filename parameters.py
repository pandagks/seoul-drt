# -*- coding: utf-8 -*-
"""
parameters.py
 - CSV 기반으로 정류장 순서/프로파일을 만들고
 - '생성 시점'에서 각 승객의 하차 정류장을 확정(항상 승차 정류장의 같은 방향의 뒤 번호)하여 Customer 리스트를 반환
 - 승차시각은 00번 기준 누적거리 × (60/speed_kmh) 만큼 뒤로 밀어 생성
"""

from typing import Dict, List, Optional
import re
import numpy as np
import pandas as pd

from customer import Customer
from route import get_distance_between, ensure_consecutive_edges

# ===== 경로/노선/차고지 기본값 (환경에 맞게 수정 가능) =====
DEFAULT_CSV_PATH = r"C:\\Users\\panda\\Desktop\\동작10전처리완료.csv"
DEFAULT_ROUTE    = "동작10"
DEPOT_ID         = "00_노들역.노량진교회"
PIVOT_STOP       = "12_상도역5번출구"   # 상·하행 기준 피벗
# =======================================================


# -----------------------------
# 유틸: 날짜 정규화/필터
# -----------------------------
def _normalize_date_str(x) -> str:
    return re.sub(r"\D", "", str(x)) if x is not None else ""

def _filter_latest_or_target_date(df: pd.DataFrame, target_date: Optional[str]):
    if "사용일자" not in df.columns:
        return df.copy(), None
    col_norm = df["사용일자"].apply(_normalize_date_str)
    used = _normalize_date_str(target_date) if target_date is not None else col_norm.max()
    return df[col_norm == used].copy(), used


# -----------------------------
# 유틸: 정류장/시간대/프로파일
# -----------------------------
def _parse_order(stop_id: str) -> int:
    m = re.match(r"^(\d+)_", str(stop_id))
    return int(m.group(1)) if m else 10**9

def _hour_cols(kind: str) -> List[str]:
    # CSV가 "13시승차수요", "13시하차수요" 형식이라고 가정
    return [f"{h}시{kind}수요" for h in range(24)]

def _present_hour_cols(df: pd.DataFrame, kind: str) -> List[str]:
    return [c for c in _hour_cols(kind) if c in df.columns]

def _default_diurnal_profile() -> np.ndarray:
    w = np.array([1,1,1,1,2,4,6,7,6,5,4,3,3,3,4,5,6,7,7,6,5,3,2,1], dtype=float)
    return w / w.sum()

def _route_hour_profile(df_any: pd.DataFrame, kind: str = "승차") -> np.ndarray:
    cols = _present_hour_cols(df_any, kind)
    if cols:
        base = df_any[cols].clip(lower=0).sum(axis=0).values.astype(float)
        s = base.sum()
        if s > 0:
            return base / s
    return _default_diurnal_profile()

def _build_stop_order(df_route_one_day: pd.DataFrame) -> List[str]:
    uniq = df_route_one_day[["정류장_ID"]].drop_duplicates().copy()
    uniq["__order"] = uniq["정류장_ID"].apply(_parse_order)
    return uniq.sort_values("__order")["정류장_ID"].tolist()


# -----------------------------
# 유틸: 누적거리(km) 계산 (거리 매트릭스 기반)
# -----------------------------
def _cumdist_along_order(stop_order: List[str]) -> Dict[str, float]:
    if not stop_order:
        raise ValueError("[parameters] stop_order가 비었습니다.")
    ensure_consecutive_edges(stop_order)  # 인접 간선 보정
    cum = {stop_order[0]: 0.0}
    for i in range(len(stop_order) - 1):
        a, b = stop_order[i], stop_order[i + 1]
        d_km = get_distance_between(a, b)
        if d_km is None:
            raise ValueError(f"[parameters] 인접 거리 누락: {a}→{b}")
        cum[stop_order[i + 1]] = cum[stop_order[i]] + d_km
    return cum


# -----------------------------
# 메인: CSV → 승차/하차/고객 생성
#   - 가능한 경우: 시간대별 실측 승차/하차를 그대로 사용
#   - 불가능(결측) 시: 총계 + 일중 프로파일로 분해
#   - 상·하행 방향 제약: PIVOT_STOP(12_상도역5번출구) 기준
#   - 승차시각: 00 기준 누적거리 × (60/speed_kmh) 분만큼 뒤로
# -----------------------------
def prepare_board_alight_from_csv(
    csv_path: str = DEFAULT_CSV_PATH,
    route: str = DEFAULT_ROUTE,
    target_date: Optional[str] = None,
    base_terminal: Optional[str] = None,
    alpha: float = 0.35,                    # 거리 감쇠 계수 (하차 분배 가중치 폴백용)
    speed_kmh: float = 20.0,                # 오프셋 계산용 속도 (km/h) → 1km=3분
    seed: int = 42,
    # --- 시간대 내 분포 옵션 ---
    ordered_per_hour: bool = True,          # 시간대 내 승차를 정류장 순서대로 분산
    ordered_spacing_min: int = 2,           # 같은 정류장/시간대 내 승객 간 분 간격
    wrap_around: bool = True,               # 정류장 순회 시 끝나면 처음으로 랩어라운드
    # --- 오프셋 상한 옵션 ---
    max_boarding_offset_min: Optional[int] = None,  # None이면 상한 없음 (3분/km 정확 반영)
):
    rng = np.random.default_rng(seed)
    df_all = pd.read_csv(csv_path)

    # 필수 컬럼 체크
    if "정류장_ID" not in df_all.columns:
        raise ValueError("CSV에 '정류장_ID' 컬럼이 없습니다.")
    if "X좌표" not in df_all.columns or "Y좌표" not in df_all.columns:
        raise ValueError("CSV에 좌표 컬럼(X좌표, Y좌표)이 필요합니다.")

    # 0) 노선 필터(있으면), 없으면 전체
    if "노선번호" in df_all.columns:
        df_route = df_all[df_all["노선번호"] == route].copy()
        if df_route.empty:
            df_route = df_all.copy()
    else:
        df_route = df_all.copy()

    # 1) 날짜 필터(없으면 최신)
    df_filt, _ = _filter_latest_or_target_date(df_route, target_date)
    if df_filt.empty and "사용일자" in df_route.columns:
        col_norm = df_route["사용일자"].apply(_normalize_date_str)
        df_filt = df_route[col_norm == col_norm.max()].copy()
    if df_filt.empty and "사용일자" in df_all.columns:
        col_norm_all = df_all["사용일자"].apply(_normalize_date_str)
        df_filt = df_all[col_norm_all == col_norm_all.max()].copy()
    if df_filt.empty:
        df_filt = df_all[["정류장_ID"]].drop_duplicates().copy()

    # 2) 정류장 단위 집계 (필요 컬럼만)
    bcols_present = _present_hour_cols(df_filt, "승차")
    acols_present = _present_hour_cols(df_filt, "하차")

    cols_present: List[str] = []
    for c in ["승차총승객수", "하차총승객수"]:
        if c in df_filt.columns:
            cols_present.append(c)
    cols_present += bcols_present + acols_present

    if cols_present:
        num_cols = [c for c in cols_present if c in df_filt.columns]
        df_day = (
            df_filt.groupby("정류장_ID", as_index=False)[num_cols]
                   .sum(numeric_only=True)
        )
    else:
        df_day = df_filt[["정류장_ID"]].drop_duplicates()

    # 총계 보강
    if "승차총승객수" not in df_day.columns:
        if bcols_present:
            board_tot = (
                df_filt.groupby("정류장_ID")[bcols_present]
                       .sum(numeric_only=True).sum(axis=1)
            )
            df_day = df_day.merge(board_tot.rename("승차총승객수"), on="정류장_ID", how="left")
        else:
            df_day["승차총승객수"] = 0
    if "하차총승객수" not in df_day.columns:
        if acols_present:
            alight_tot = (
                df_filt.groupby("정류장_ID")[acols_present]
                       .sum(numeric_only=True).sum(axis=1)
            )
            df_day = df_day.merge(alight_tot.rename("하차총승객수"), on="정류장_ID", how="left")
        else:
            df_day["하차총승객수"] = 0

    for c in ["승차총승객수", "하차총승객수"]:
        if c in df_day.columns:
            df_day[c] = df_day[c].fillna(0).astype(int)

    # 3) 정류장 순서/누적거리
    stop_order = _build_stop_order(df_day)
    if len(stop_order) == 0:
        empty = pd.DataFrame(columns=[f"{h}시승차수요" for h in range(24)])
        return [], empty, empty.copy(), []
    cumdist = _cumdist_along_order(stop_order)

    # 3-1) 상행/하행 split
    if PIVOT_STOP not in stop_order:
        raise ValueError(f"[parameters] 피벗 정류장({PIVOT_STOP})이 stop_order에 없습니다.")
    pivot_idx = stop_order.index(PIVOT_STOP)
    up_stops   = stop_order[:pivot_idx + 1]      # 00 ~ 12
    down_stops = stop_order[pivot_idx + 1:]      # 13 ~

    # 4) 승차 테이블 (정류장×24)
    hcols_b = [f"{h}시승차수요" for h in range(24)]
    if bcols_present:
        # 실측 시간대별 승차를 그대로 사용
        board_df = (
            df_filt.groupby("정류장_ID")[bcols_present]
                   .sum(numeric_only=True)
                   .reindex(stop_order)
                   .fillna(0)
                   .astype(int)
        )
        # 누락 시간대 0 채움 + 컬럼 정렬
        for c in hcols_b:
            if c not in board_df.columns:
                board_df[c] = 0
        board_df = board_df[hcols_b]
    else:
        # 시간대가 없으면 총계를 일중 프로파일로 분배
        prof = _route_hour_profile(df_filt, kind="승차")
        rows = []
        for _, r in df_day.iterrows():
            T = int(max(0, r.get("승차총승객수", 0)))
            by_h = list(rng.multinomial(T, prof)) if T > 0 else [0] * 24
            row = {"정류장_ID": r["정류장_ID"]}
            row.update({h: int(c) for h, c in zip(hcols_b, by_h)})
            rows.append(row)
        board_df = (
            pd.DataFrame(rows)
            .set_index("정류장_ID")
            .reindex(stop_order)
            .fillna(0)
            .astype(int)
        )

    # 5) 하차 테이블 (정류장×24) — 실측이 있으면 참고용으로 로드
    hcols_a = [f"{h}시하차수요" for h in range(24)]
    if acols_present:
        alight_ref = (
            df_filt.groupby("정류장_ID")[acols_present]
                   .sum(numeric_only=True)
                   .reindex(stop_order)
                   .fillna(0)
                   .astype(int)
        )
        for c in hcols_a:
            if c not in alight_ref.columns:
                alight_ref[c] = 0
        alight_ref = alight_ref[hcols_a]
    else:
        # 실측 하차가 없을 때를 대비한 0 테이블(참조용)
        alight_ref = pd.DataFrame(0, index=stop_order, columns=hcols_a, dtype=int)

    # 실제 생성된 고객 기반 하차 카운트를 담을 표(검증/리포트용)
    alight_df = pd.DataFrame(0, index=stop_order, columns=hcols_a, dtype=int)

    # 6) 고객 생성(같은 방향 downstream으로 하차 확정, 승차시각 오프셋 반영)
    first_stop = stop_order[0]
    if base_terminal is None:
        base_terminal = first_stop

    customers: List[Customer] = []
    cid = 0
    base0 = cumdist.get(base_terminal, cumdist[first_stop])

    for i_idx, i in enumerate(stop_order):
        # 같은 '방향'의 다운스트림만 하차 후보
        if i in up_stops:
            # 상행: i 뒤~pivot까지
            downstream = [s for j, s in enumerate(stop_order) if (j > i_idx) and (j <= pivot_idx)]
        else:
            # 하행: i 뒤~마지막까지
            downstream = [s for j, s in enumerate(stop_order) if (j > i_idx)]

        if not downstream:
            continue  # 마지막이거나 같은 방향 다운스트림 없음

        # 거리 기반 폴백 가중치 준비 (멀수록 확률↓)
        dists = np.array([max(0.0, cumdist[d] - cumdist[i]) for d in downstream], dtype=float)
        dist_w = np.exp(-alpha * dists)
        dist_w = dist_w / dist_w.sum()

        # 정류장별 오프셋(기점→해당 정류장 이동 시간)
        raw_offset_min = int((max(0.0, cumdist[i] - base0) / max(1e-6, speed_kmh)) * 60)  # 1km=3분
        if max_boarding_offset_min is None:
            base_offset_min = raw_offset_min
        else:
            base_offset_min = min(raw_offset_min, int(max_boarding_offset_min))

        for h in range(24):
            b = int(board_df.at[i, f"{h}시승차수요"]) if i in board_df.index else 0
            if b <= 0:
                continue

            # 같은 시간대의 하차 실측 비율이 있으면 우선 사용, 없으면 거리 폴백
            if acols_present:
                w = alight_ref.loc[downstream, f"{h}시하차수요"].astype(float).to_numpy()
                if w.sum() > 0:
                    weights = w / w.sum()
                else:
                    weights = dist_w
            else:
                weights = dist_w

            cnts = np.random.multinomial(b, weights) if b > 0 else np.zeros_like(weights, dtype=int)

            # 같은 정류장/시간대 내 승객 간 분 단위 간격 부여
            minute_cursor = h * 60 + base_offset_min
            for d_stop, cnt in zip(downstream, cnts.tolist()):
                if cnt <= 0:
                    continue
                for _ in range(int(cnt)):
                    customers.append(
                        Customer(
                            customer_id=cid,
                            boarding_stop=i,
                            getoff_stop=d_stop,  # 같은 방향 downstream으로 하차 확정
                            time=minute_cursor   # 00기준 누적거리 × 3분/km 반영
                        )
                    )
                    cid += 1
                    minute_cursor += max(0, int(ordered_spacing_min))
                # 생성된 고객 기반 하차 카운트(승차 시간대 h에 귀속)
                alight_df.at[d_stop, f"{h}시하차수요"] += int(cnt)

    # 인덱스 이름 정리
    board_df.index.name = "정류장_ID"
    alight_df.index.name = "정류장_ID"

    # 반환
    return customers, board_df, alight_df, stop_order


# -----------------------------
# 시뮬레이터 호환 로더
# -----------------------------
def load_fixed_customers(
    csv_path: str = DEFAULT_CSV_PATH,
    route: str = DEFAULT_ROUTE,
    target_date: Optional[str] = None,
    base_terminal: Optional[str] = None,
    alpha: float = 0.35,
    speed_kmh: float = 20.0,
    seed: int = 42,
    ordered_per_hour: bool = True,
    ordered_spacing_min: int = 2,
    wrap_around: bool = True,
    max_boarding_offset_min: Optional[int] = None,   # None: 상한 없음
):
    """
    시뮬레이터에서 바로 쓰는 고정 고객 생성기.
    - 생성 시점에서 하차 정류장을 확정(항상 '같은 방향'의 승차 뒤 번호)
    - 승차시각은 00 기준 누적거리 × (60/speed_kmh)만큼 뒤로 밀림
    - 시간대별 실측 승/하차가 있으면 그대로 이용, 없으면 총계+프로파일로 분배
    """
    customers, _, _, _ = prepare_board_alight_from_csv(
        csv_path=csv_path,
        route=route,
        target_date=target_date,
        base_terminal=base_terminal or DEPOT_ID,
        alpha=alpha,
        speed_kmh=speed_kmh,
        seed=seed,
        ordered_per_hour=ordered_per_hour,
        ordered_spacing_min=ordered_spacing_min,
        wrap_around=wrap_around,
        max_boarding_offset_min=max_boarding_offset_min,
    )
    return customers
