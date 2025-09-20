import streamlit as st
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io, sys
import koreanize_matplotlib
from simulator import Simulation  # DRT 시뮬레이터
from visualization_route import plot_route  # 시각화 함수
from streamlit_folium import st_folium

st.set_page_config(layout="wide")

# 노선 옵션 및 파일 경로 설정
route_options = {
    '25번': {
        'demand_file': "C:\\Users\\panda\\Documents\\졸작\\result\\bus_25(10-16).xlsx",
        'dropoff_file': "C:\\Users\\panda\\Documents\\졸작\\result\\25번_정류장_승하차\\승하차정류장_ID.csv",
        'coord_file': "C:\\Users\\panda\\Documents\\졸작\\result\\정류장_좌표.xlsx"
    },
    '23번': {
        'demand_file': "C:\\Users\\panda\\Documents\\졸작\\result\\bus_23(10-16).xlsx",
        'dropoff_file': "C:\\Users\\panda\\Documents\\졸작\\result\\25번_정류장_승하차\\승하차정류장_ID.csv",
        'coord_file': "C:\\Users\\panda\\Documents\\졸작\\result\\정류장_좌표.xlsx"
    }
}

시간대들 = ['10', '11', '12', '13', '14', '15', '16']
승차컬럼 = [f"{h}(승차)" for h in 시간대들]
하차컬럼 = [f"{h}(하차)" for h in 시간대들]

@st.cache_data
def load_demand_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)[['정류장_ID', '일'] + 시간대들]

@st.cache_data
def load_dropoff_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_coord_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def generate_prediction_local(df, target_date):
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
        st.warning(f"[경고] 일자 '{date_str}'에 해당하는 데이터가 없습니다.")
        return pd.DataFrame()

    np.random.seed(hash(target_date) % (2**32))

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

def plot_tradeoff_curve():
    x = np.linspace(0, 1, 100)
    cost_from_distance = 10000 + 8000 * np.exp(4 * (x - 1))
    cost_from_wait = 10000 + 8000 * np.exp(-4 * x)
    mid_index = np.argmin(np.abs(cost_from_distance - cost_from_wait))
    opt_x = x[mid_index]
    opt_cost = cost_from_distance[mid_index]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, cost_from_distance, label="거리 vs 비용", color='blue')
    ax.plot(x, cost_from_wait, label="대기시간 vs 비용", color='orange')
    ax.plot(opt_x, opt_cost, 'ro', label="최적 Trade-off 점")

    ax.set_xlabel("비중 (거리: 0 → 대기시간: 1)", fontsize=12)
    ax.set_ylabel("예상 비용 (원)", fontsize=12)
    ax.set_title("거리 기반과 대기시간 기반 비용 곡선의 교차", fontsize=14)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

def run_simulation_for_route(route_key):
    if 'simulations' not in st.session_state:
        st.session_state.simulations = {}
    if route_key not in st.session_state.simulations:
        sim = Simulation()
        st.session_state.simulations[route_key] = sim.run()
    return st.session_state.simulations[route_key]

import streamlit as st

def main():
    if "menu_shown" not in st.session_state:
        st.session_state.menu_shown = False

    if not st.session_state.menu_shown:
        st.title("🚍 DRT 시뮬레이션 및 시각화")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(
                """
                <div style="text-align: left;">
                    <h2>팀원 소개</h2>
                    <hr style="border:1px solid #999999;">
                    <div style="
                        border: 2px solid #555555;
                        border-radius: 8px;
                        padding: 10px;
                        background-color: #f0f0f0;
                        max-width: 250px;
                    ">
                        <ul style="margin:0; padding-left: 20px;">
                            <li>한승훈</li>
                            <li>한현성</li>
                            <li>송도훈</li>
                            <li>최승환</li>
                            <li>이현규</li>
                        </ul>
                    </div>
                    <hr>
                    <p>아래 버튼을 클릭하면 시뮬레이션 및 시각화 기능으로 이동합니다.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 버튼 따로 스트림릿으로 빼서 작동하게
            if st.button("시작하기 ▶"):
                st.session_state.menu_shown = True
                return

        with col2:
            st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
            st.image("D:/downloads/DRT_benor.jpg", width=550)
            st.markdown('</div>', unsafe_allow_html=True)

        return

    menu = st.sidebar.selectbox(
        "Menu", 
        ["📊 예측 및 히트맵 시각화", "▶ 시뮬레이션 실행", "📋 결과 확인", "🗺️ 노선 경로 시각화"]
    )

    if menu == "📊 예측 및 히트맵 시각화":
        st.title("🚍 DRT 수요 예측 및 하차 히트맵 시각화")
        selected_route = st.selectbox("버스 노선을 선택하세요", list(route_options.keys()))
        file_paths = route_options[selected_route]
        target_date = st.date_input("예측할 날짜를 선택하세요", value=datetime.date(2024, 3, 4))

        df_demand = load_demand_data(file_paths['demand_file'])
        df_dropoff = load_dropoff_data(file_paths['dropoff_file'])

        predicted = generate_prediction_local(df_demand, target_date)

        tab1, tab2, tab3 = st.tabs(["수요 분포 추정값", "하차 히트맵", "거리 vs 대기시간 Trade-off"])

        with tab1:
            st.subheader("📈 수요 분포 추정값")
            if not predicted.empty:
                st.dataframe(predicted)
                sum_by_hour = predicted[승차컬럼].sum()
                st.bar_chart(sum_by_hour)
            else:
                st.warning("선택한 날짜에 데이터가 없습니다.")

        with tab2:
            st.subheader("📍 하차 히트맵")
            raw_df = df_dropoff[['정류장_ID'] + 하차컬럼].set_index('정류장_ID')
            norm = df_dropoff.set_index('정류장_ID')['통과노선수']
            heatmap_df = raw_df.div(norm, axis=0).fillna(0)

            fig, ax = plt.subplots(figsize=(10, max(4, len(heatmap_df)*0.25)))
            sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
            st.pyplot(fig)

        with tab3:
            st.subheader("🚍 거리 vs 대기시간 Trade-off 시각화")
            plot_tradeoff_curve()

    elif menu == "▶ 시뮬레이션 실행":
        st.header("▶ 시뮬레이션 실행하기")
        st.markdown("시뮬레이션 시작버튼을 눌러주세요.")

        if st.button("시뮬레이션 시작"):
            log_buffer = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = log_buffer
            plt.close("all")

            sim = Simulation()
            sim.run()

            sys.stdout = old_stdout

            log_text = log_buffer.getvalue()
            st.session_state["last_run_log"] = log_text
            st.session_state["last_run_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.success("✅ 시뮬레이션이 완료되었습니다.  \n'결과 확인' 메뉴로 이동해주세요.")

    elif menu == "📋 결과 확인":
        st.header("📊 시뮬레이션 결과 확인")

        if "last_run_log" not in st.session_state:
            st.warning("아직 시뮬레이션이 실행되지 않았습니다.  \n'시뮬레이션 실행' 메뉴에서 실행해주세요.")
            return

        st.subheader("📝 시뮬레이션 로그")
        st.code(st.session_state["last_run_log"])

        fig_nums = plt.get_fignums()
        if not fig_nums:
            st.info("노선 경로를 확인하시려면 '노선 경로 시각화'로 이동해주세요.")
        else:
            for num in fig_nums:
                fig = plt.figure(num)
                st.subheader(f"📈 Figure {num}")
                st.pyplot(fig)
            plt.close("all")

    elif menu == "🗺️ 노선 경로 시각화":
        st.title("🗺️ 노선 및 시간대별 경로 시각화")

        selected_route = st.selectbox("노선을 선택하세요", list(route_options.keys()))
        coord_df = load_coord_data(route_options[selected_route]['coord_file'])

        # 시뮬레이션 결과 가져오기
        total_route = run_simulation_for_route(selected_route)
        selected_hour = st.selectbox("시간대를 선택하세요", options=sorted(total_route.keys()))

        # 경로 데이터 생성
        df_selected = coord_df
        plot_df = pd.DataFrame()
        for station in total_route[selected_hour]: 
            station_id = station.split('(')[0]
            plot_df = pd.concat([plot_df, df_selected[df_selected['정류장_ID'] == station_id]], axis=0)

        # 지도 생성
        m = plot_route(plot_df)

        # --- 화면 2분할 ---
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🚌 기존 노선")
            st.image("C:\\vscode\\졸작\\25번원래노선.png", use_container_width=True)

        with col2:
            st.subheader("🛠 최적화된 노선")
            st_folium(m, width=700, height=500)

        # --- 거리 비교 ---
        from ga_optimizer import evaluate_sequence
        from utils import get_distance_between

        # 시뮬레이션 거리 계산
        unique_seq = []
        seen = set()
        for stop in [s.split('(')[0] for s in total_route[selected_hour]]:
            if stop not in seen:
                unique_seq.append(stop)
                seen.add(stop)

        optimized_distance = 0
        for i in range(len(unique_seq) - 1):
            dist = get_distance_between(unique_seq[i], unique_seq[i+1])
            if dist:
                optimized_distance += dist

        # 차고지 복귀 거리 추가
    
        return_dist = get_distance_between(unique_seq[-1], "00_오이도차고지")
        if return_dist:
            optimized_distance += return_dist

        st.markdown(f"**📏 기존 노선 거리:** 22.10 km")
        st.markdown(f"**📏 최적화된 노선 거리 ({selected_hour}시):** {optimized_distance:.2f} km")


if __name__ == "__main__":
    main()