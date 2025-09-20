import heapq
import pandas as pd
import numpy as np
from bus import Bus
from route import get_distance_between

# 고정된 고객 목록을 반환
def generate_daily_poisson_prediction(df, target_date, seed=42):
    시간대들 = ['10', '11', '12', '13', '14', '15', '16']
    df['월'] = pd.to_datetime(df['일']).dt.month
    train_df = df[df['월'].between(3, 10)]

    stats = (
        train_df
        .groupby('정류장_ID')[시간대들]
        .agg(['mean', 'std'])
    )
    stats.columns = [f'{col}_{stat}' for col, stat in stats.columns]
    stats = stats.reset_index()

    date_str = pd.to_datetime(target_date).strftime('%Y-%m-%d')
    test_df = df[df['일'] == date_str].copy()
    if test_df.empty:
        print(f"[경고] 일자 '{date_str}'에 해당하는 데이터가 없습니다.")
        return pd.DataFrame()

    np.random.seed(seed)
    결과 = []
    for _, row in test_df.iterrows():
        정류장 = row['정류장_ID']
        통계행 = stats[stats['정류장_ID'] == 정류장]
        if 통계행.empty:
            continue
        예측 = {'정류장_ID': 정류장, '일': date_str}
        for 시간 in 시간대들:
            λ = 통계행[f'{시간}_mean'].values[0]
            λ = max(λ, 1e-6)
            예측[f'{시간}(승차)'] = int(np.random.poisson(λ))
        결과.append(예측)

    return pd.DataFrame(결과)

def get_dropoff_distribution(df, 승차정류장, 시간대, 수요수):
    시간컬럼 = f"{시간대}(하차)"
    try:
        index_start = df[df["정류장_ID"] == 승차정류장].index[0]
    except IndexError:
        return [승차정류장] * 수요수

    after_df = df.iloc[index_start+1:].copy()
    after_df = after_df[after_df[시간컬럼].notna()]
    if after_df.empty:
        return [승차정류장] * 수요수

    probs = after_df[시간컬럼].values
    probs = probs / probs.sum()

    return np.random.choice(after_df["정류장_ID"], size=수요수, p=probs)


from customer import Customer
from utils import get_distance_between
import pandas as pd
import numpy as np
import random

def load_fixed_customers():
    df_수요 = pd.read_excel("C:\\Users\\panda\\Documents\\졸작\\result\\bus_25(10-16).xlsx")[['정류장_ID', '일', '10', '11', '12', '13', '14', '15', '16']]
    df_하차비율 = pd.read_csv("C:\\Users\\panda\\Documents\\졸작\\result\\25번_정류장_승하차\\승하차정류장_ID.csv")
    시간대들 = ['10', '11', '12', '13', '14', '15', '16']

    # 수요 예측
    df_수요예측 = generate_daily_poisson_prediction(df_수요, target_date="2024-11-15")

    # 정류장별 기준 시각 계산 (00번 기준 거리 기반)
    기준정류장 = "00_노들역.노량진교회"
    정류장목록 = df_수요예측['정류장_ID'].tolist()
    정류장별_시간 = {}
    for 정류장 in 정류장목록:
        거리 = get_distance_between(기준정류장, 정류장)
        거리 = 거리 if 거리 is not None else 0
        시간_분 = int(거리 * 3)
        정류장별_시간[정류장] = 시간_분

    customers = []
    customer_id = 0

    for _, row in df_수요예측.iterrows():
        승차정류장 = row["정류장_ID"]
        for 시간 in 시간대들:
            수요수 = int(row.get(f"{시간}(승차)", 0))
            if 수요수 == 0:
                continue

            # 하차 정류장 분포 기반 생성
            하차후보들 = get_dropoff_distribution(df_하차비율, 승차정류장, 시간, 수요수)

            # 승차 시각 계산
            기준_시 = int(시간)
            거리기반_분 = 정류장별_시간.get(승차정류장, 0)
            시 = 기준_시 + (거리기반_분 // 60)
            분 = 거리기반_분 % 60

            for i in range(수요수):
                customers.append(Customer(
                    customer_id=customer_id,
                    boarding_stop=승차정류장,
                    getoff_stop=하차후보들[i],
                    time=시 * 60 + 분  # 분 단위
                ))
                customer_id += 1

    return customers

