from route import load_distance_matrix_csv
from parameters import prepare_board_alight_from_csv, load_fixed_customers

# 1) 가장 먼저 거리 매트릭스 로드 (미터 → km 변환)
load_distance_matrix_csv("D:\\jupyter_env\\projects\\drt\\distance_matrix.csv", unit="m")  # 내부 km 저장

# 2) 생성 (route가 맞는지 확신이 없으면 route=None으로 전체 사용도 가능)
CSV_PATH = "C:\\Users\\panda\\Desktop\\동작10전처리완료.csv"
customers, board_df, alight_df, stop_order = prepare_board_alight_from_csv(
    csv_path=CSV_PATH,
    route="동작10",          # 필요시 "동작10" 대신 None로 시도 가능
    target_date=None,        # 미지정 시 최신일자 자동 선택
    base_terminal=None,
    alpha=0.35,
    speed_kmh=20.0,
    seed=42
)

# 기존 인터페이스
CUSTOMERS = load_fixed_customers(
    csv_path=CSV_PATH, route="동작10",
    target_date=None, base_terminal=stop_order[0],
    alpha=0.35, speed_kmh=20.0, seed=42
)


# 2-1) 넓은 포맷(정류장×24시간)
board_df.to_csv("generated_board_wide.csv", encoding="utf-8-sig")
alight_df.to_csv("generated_alight_wide.csv", encoding="utf-8-sig")

# 2-2) 롱 포맷(정류장,시각,수요)
def _to_long(df, kind):
    tmp = df.copy()
    tmp.index.name = "정류장_ID"
    out = tmp.reset_index().melt(id_vars=["정류장_ID"],
                                 var_name="시각", value_name=f"{kind}수요")
    # "13시승차수요" → 13
    out["hour"] = out["시각"].str.extract(r"(\d+)").astype(int)
    return out[["정류장_ID", "hour", f"{kind}수요"]]

board_long = _to_long(board_df, "승차")
alight_long = _to_long(alight_df, "하차")

board_long.to_csv("generated_board_long.csv", index=False, encoding="utf-8-sig")
alight_long.to_csv("generated_alight_long.csv", index=False, encoding="utf-8-sig")
print("✅ 저장 완료: generated_*_*.csv")

# basic.py에 추가
import numpy as np, math
from parameters import _hour_cols  # 이미 있다면 재import 불필요

def attach_getoffs_for_customers(customers, stop_order, cumdist, alpha=0.35, speed_kmh=20.0, seed=42):
    """승차 정류장 이후 다운스트림으로 확률 분배(가까울수록 확률↑) + 이동시간 반영"""
    rng = np.random.default_rng(seed)
    # 시간대별 하차 카운트(검증용)
    hcols_a = _hour_cols("하차")
    alight_by_stop_hour = {s: {h:0 for h in range(24)} for s in stop_order}

    # 정류장→index
    idx = {s:i for i,s in enumerate(stop_order)}

    for c in customers:
        i = idx[c.boarding_stop]
        downstream = stop_order[i+1:] or [c.boarding_stop]
        dists = np.array([max(0.0, cumdist[d]-cumdist[c.boarding_stop]) for d in downstream], dtype=float)
        weights = np.exp(-alpha * dists) if len(dists) > 0 else np.array([1.0])
        weights = weights / weights.sum()
        pick = rng.choice(len(downstream), p=weights)
        dest = downstream[pick]
        c.getoff_stop = dest

        # 도착 시각(hour) 계산(대략)
        dist_km = dists[pick] if len(dists)>0 else 0.0
        travel_min = 0 if speed_kmh <= 0 else int(math.ceil((dist_km / speed_kmh) * 60.0))
        arrive_h = min(23, (c.time // 60) + (travel_min // 60))
        alight_by_stop_hour[dest][arrive_h] += 1

    # 검증: 고객 기반 하차 합계
    total = sum(sum(h.values()) for h in alight_by_stop_hour.values())
    print(f"attach_getoffs: 총 하차수 = {total} (고객수={len(customers)})")
    return customers, alight_by_stop_hour

# 사용 예
# cumdist는 parameters.prepare_board_alight_from_csv 내부에서 썼던 누적거리 dict가 필요
# 없으면 간단히 다시 만들자:
from route import get_distance_between, ensure_consecutive_edges
def _recompute_cumdist(stop_order):
    ensure_consecutive_edges(stop_order)
    out = {stop_order[0]: 0.0}
    for i in range(len(stop_order)-1):
        d = get_distance_between(stop_order[i], stop_order[i+1])
        out[stop_order[i+1]] = out[stop_order[i]] + d
    return out

cumdist = _recompute_cumdist(stop_order)
customers, al_by = attach_getoffs_for_customers(customers, stop_order, cumdist, alpha=0.35, speed_kmh=20.0, seed=42)

# demand_dump.py (새 파일로)
from route import load_distance_matrix_csv
from parameters import prepare_board_alight_from_csv
import pandas as pd

# 1) 거리 매트릭스 로드
load_distance_matrix_csv(r"D:\\jupyter_env\\projects\\drt\\distance_matrix.csv", unit="m")

# 2) 파라미터에서 승/하차 테이블과 고객 생성
CSV_PATH = r"C:\\Users\\panda\\Desktop\\동작10전처리완료.csv"
customers, board_df, alight_df, stop_order = prepare_board_alight_from_csv(
    csv_path=CSV_PATH, route="동작10", target_date=None,
    base_terminal=None, alpha=0.35, speed_kmh=20.0, seed=42
)

# 3) 승/하차(정류장×24) 저장
board_df.to_csv("generated_board_wide.csv", encoding="utf-8-sig")
alight_df.to_csv("generated_alight_wide.csv", encoding="utf-8-sig")

# 4) 롱포맷(정류장, hour, 수요) 저장
def _to_long(df, kind):
    tmp = df.copy()
    tmp.index.name = "정류장_ID"
    out = tmp.reset_index().melt(id_vars=["정류장_ID"], var_name="시각", value_name=f"{kind}수요")
    out["hour"] = out["시각"].str.extract(r"(\d+)").astype(int)
    return out[["정류장_ID", "hour", f"{kind}수요"]]

_to_long(board_df, "승차").to_csv("generated_board_long.csv", index=False, encoding="utf-8-sig")
_to_long(alight_df, "하차").to_csv("generated_alight_long.csv", index=False, encoding="utf-8-sig")

# 5) (옵션) 시뮬레이터가 사용하는 customers로부터 실제 '시간대별 승차' 재집계
cust = pd.DataFrame([{"stop": c.boarding_stop, "hour": c.time // 60} for c in customers])
cust_pivot = (
    cust.groupby(["stop", "hour"])
        .size()                         # 건수 카운트
        .unstack(fill_value=0)          # hour 열들로 피벗
        .sort_index()
)
cust_pivot.to_csv("customers_board_by_hour.csv", encoding="utf-8-sig")
print("✅ 저장 완료: customers_board_by_hour.csv")
