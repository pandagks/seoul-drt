# simulator.py
# -*- coding: utf-8 -*-

from statistics import mean, stdev
import numpy as np

from parameters import load_fixed_customers, DEPOT_ID  # DEPOT_ID = "00_노들역.노량진교회"
from customer import Customer
from route import get_distance_between, load_distance_matrix_csv
from bus import Bus
from ga_optimizer import run_ga

# =========================
# 설정
# =========================
# 거리 매트릭스 로드(경로/단위는 환경에 맞게)
try:
    load_distance_matrix_csv(r"D:\\jupyter_env\\projects\\drt\\distance_matrix.csv", unit="m")
except Exception:
    load_distance_matrix_csv("distance_matrix.csv", unit="m")

DEPOT = DEPOT_ID if 'DEPOT_ID' in globals() else "00_노들역.노량진교회"
BUS_CAPACITY = 40            # 버스 정원
WAIT_LIMIT_MIN = 45          # 승객 포기 기준(분)
SPEED_MIN_PER_KM = 3         # 이동시간 환산(분/km)
MAX_BUSES = 2                # 최대 2대까지만
RETURN_GAP_MIN = 25          # 시간 경계에서 복귀 임계 gap(분)
NEXT_HOUR_DEMAND_TH = 5      # 다음 시간대 수요가 이 미만이면 복귀

def calculate_cost(total_distance_km, cost_per_km=3000):
    return int(round(total_distance_km * cost_per_km))

# =========================
# 메인 시뮬레이터
# =========================
class Simulation:
    def __init__(self):
        self.customers = []
        self.waiting_customers = {}   # {stop: [Customer, ...]}
        self.buses = []
        self.bus_counter = 0
        self.current_time = 0         # 분, 00:00=0
        self.total_distance_across_runs = 0.0
        self.total_time_across_runs = 0
        self.fitness_all = []
        self.abandoned_customers = 0

    def generate_customers(self):
        # parameters.py에서 생성 시점에 하차까지 확정
        fixed_customers = load_fixed_customers(
            ordered_per_hour=True,
            ordered_spacing_min=2,
            wrap_around=True,
            max_boarding_offset_min=None  # 00→정류장 누적거리 × 3분/km를 정확히 반영
        )
        for c in fixed_customers:
            if c.boarding_stop != c.getoff_stop:
                self.customers.append(c)
                self.waiting_customers.setdefault(c.boarding_stop, []).append(c)
        # 승객 도착시각 순으로 정렬
        self.customers.sort(key=lambda x: x.time)

    def _simulate_bus_run(self, bus, path, hour_min=None, hour_end=None):
        """
        경로(path)를 순회하면서
        - 하차 먼저
        - 승차: '도착해 기다리고 있는' 승객만 시간순으로 탑승
        - 아직 도착하지 않은 승객(c.time > current_time)은 건드리지 않음
        """
        for i, stop in enumerate(path):
            # 이동
            if i > 0:
                prev = path[i-1]
                d = get_distance_between(prev, stop) or 0.0
                move = int(d * SPEED_MIN_PER_KM)
                bus.total_distance += d
                self.current_time += move
            bus.current_stop = stop

            # 1) 하차
            dropped = bus.drop_customer(stop, self.current_time)
            for c in dropped:
                c.dropoff_time = self.current_time

            # 2) 승차 (도착해 기다리는 사람만)
            if stop not in self.waiting_customers:
                continue

            # 시간순 정렬
            waiting_list = sorted(self.waiting_customers[stop], key=lambda x: x.time)

            keep_list = []
            for c in waiting_list:
                # 아직 정류장에 도착하지 않은 승객은 보류
                if c.time > self.current_time:
                    keep_list.append(c)
                    continue

                wait = self.current_time - c.time  # ✅ '대기 시작 시각'부터의 실제 대기시간

                # 너무 오래 기다려서 포기
                if wait > WAIT_LIMIT_MIN:
                    self.abandoned_customers += 1
                    # 리스트에서 제외(버스가 와도 안 탐)
                    continue

                # 좌석 여유 있으면 탑승
                if bus.can_board_customer():
                    bus.board_customer(c, self.current_time)
                    c.pickup_time = self.current_time
                    # 탑승자는 대기열에서 제거
                else:
                    # 좌석이 없으면 나머지는 일단 유지
                    keep_list.append(c)

            # 정류장 대기열 갱신
            self.waiting_customers[stop] = keep_list


    def _force_sweep_within_hour(self, hourly_customers, hour_min, hour_end):
        """
        한 시간 내(예: 11:00~11:59)에 생성된 승객이 그 시간 안에 반드시 하차하도록
        시간대 종료 시점에 온보드 고객/미탑승 고객을 강제 처리한다.
        - 온보드 고객: 현재 버스 위치에서 해당 하차정류장까지 즉시 이동하여 하차 처리
        - 미탑승 고객: 해당 시간대 내 픽업 및 하차를 즉시 처리
        """
        if not hourly_customers:
            return

        # 1) 각 버스의 탑승자 강제 하차
        for bus in self.buses:
            if not bus.onboard_customers:
                continue
            for cust in list(bus.onboard_customers):
                d = get_distance_between(bus.current_stop, cust.getoff_stop) or 0.0
                t = int(d * SPEED_MIN_PER_KM)
                # 이동
                bus.total_distance += d
                self.current_time += t
                # 시간대 안으로 클램핑
                if self.current_time >= hour_end:
                    self.current_time = hour_end - 1
                # 정류장 도착 및 하차 처리
                bus.current_stop = cust.getoff_stop
                bus.onboard_customers.remove(cust)
                bus.finished_customers.append((cust, self.current_time))
                cust.dropoff_time = self.current_time
                # pickup_time이 없다면 이 시점에 픽업된 것으로 간주
                if not hasattr(cust, "pickup_time"):
                    cust.pickup_time = max(hour_min, min(self.current_time - 1, hour_end - 1))

        # 2) 아직 탑승하지 못한(대기열에 남은) 시간대 승객 강제 처리
        remaining = []
        hourly_ids = {c.customer_id for c in hourly_customers}
        for stop, lst in list(self.waiting_customers.items()):
            keep = []
            for c in lst:
                if c.customer_id in hourly_ids:
                    remaining.append((stop, c))
                else:
                    keep.append(c)
            self.waiting_customers[stop] = keep

        # 강제 픽업 & 하차 (시간대 내부에서 처리)
        for stop, cust in remaining:
            pickup_t = max(hour_min, min(self.current_time, hour_end - 2))
            cust.pickup_time = pickup_t
            # 하차 이동 처리
            d = get_distance_between(stop, cust.getoff_stop) or 0.0
            t = int(d * SPEED_MIN_PER_KM)
            self.current_time = max(self.current_time, pickup_t) + t
            if self.current_time >= hour_end:
                self.current_time = hour_end - 1
            cust.dropoff_time = self.current_time
            # 버스 거리 총합엔 반영(가장 가까운 버스에 얹었다고 가정)
            if self.buses:
                self.buses[0].total_distance += d

    def _maybe_return_to_depot(self, hour, hour_end):
        """시간 경계에서 gap이 크고 다음 시간대 수요가 적으면 복귀."""
        next_hour_min = (hour + 1) * 60 if hour < 23 else None
        if next_hour_min is None:
            return
        if self.current_time < hour_end:
            self.current_time = hour_end
        gap = next_hour_min - self.current_time
        next_count = len([c for c in self.customers if next_hour_min <= c.time < next_hour_min + 60])
        if gap >= RETURN_GAP_MIN and next_count < NEXT_HOUR_DEMAND_TH:
            for bus in self.buses:
                if bus.current_stop != DEPOT:
                    d = get_distance_between(bus.current_stop, DEPOT) or 0.0
                    if d > 0:
                        bus.total_distance += d
                        self.current_time += int(d * SPEED_MIN_PER_KM)
                        bus.current_stop = DEPOT

    def run(self):
        self.generate_customers()

        # 초기 1대 투입
        self.bus_counter += 1
        self.buses = [Bus(bus_id=f"Bus{self.bus_counter}", current_stop=DEPOT, max_capacity=BUS_CAPACITY)]

        total_route_summary = {}

        # 하루 24개 사이클(0~23시)
        for hour in range(0, 24):
            hour_min = hour * 60
            hour_end = hour_min + 60
            if self.current_time < hour_min:
                self.current_time = hour_min

            # 이 시간대 고객
            hourly_customers = [c for c in self.customers if hour_min <= c.time < hour_end]
            if not hourly_customers:
                print(f"[{hour:02d}시] 수요 없음")
                self._maybe_return_to_depot(hour, hour_end)
                continue

            # ✅ 최초 수요 시각까지 차고지 대기 후 출발
            first_ready = min(c.time for c in hourly_customers)
            if self.current_time < first_ready:
                hh, mm = divmod(first_ready, 60)
                print(f"[{hour:02d}시] 최초 수요 {hh:02d}:{mm:02d} → 차고지 대기 후 출발")
                self.current_time = first_ready

            # GA로 경로 생성(최적화식 유지)
            pairs = [(c.boarding_stop, c.getoff_stop) for c in hourly_customers]
            stops_to_visit, fitness_with_return, distance, minutes = run_ga(pairs, verbose=False)
            self.fitness_all.extend(fitness_with_return)
            self.total_distance_across_runs += distance
            self.total_time_across_runs += minutes

            # 경로 요약(승/하차 수)
            board_cnt, drop_cnt = {}, {}
            for c in hourly_customers:
                board_cnt[c.boarding_stop] = board_cnt.get(c.boarding_stop, 0) + 1
                drop_cnt[c.getoff_stop] = drop_cnt.get(c.getoff_stop, 0) + 1
            labels = []
            for s in stops_to_visit:
                info = []
                if board_cnt.get(s, 0): info.append(f"{board_cnt[s]}승차")
                if drop_cnt.get(s, 0): info.append(f"{drop_cnt[s]}하차")
                labels.append(s if not info else f"{s}({', '.join(info)})")
            print(f"[{hour:02d}시 사이클] 경로: {' → '.join(labels)}")
            total_route_summary[str(hour)] = labels

            # 시간대 이동거리 합 계산용 스냅샷
            start_dist = {b.bus_id: b.total_distance for b in self.buses}
            aband_start = self.abandoned_customers

            # ----- 1) Bus1 운행 -----
            self._simulate_bus_run(self.buses[0], stops_to_visit, hour_min, hour_end)

            # 이 시간대 미탑승 고객이 남았는지 판단
            served_ids_bus1 = {cust.customer_id for cust, _ in self.buses[0].finished_customers}
            remaining_hourly = [
                c for c in hourly_customers
                if (not hasattr(c, "pickup_time")) and (c.customer_id not in served_ids_bus1)
            ]

            # ----- 2) 필요 시 Bus2 한 번만 추가 투입 -----
            if remaining_hourly and len(self.buses) < MAX_BUSES:
                self.bus_counter += 1
                self.buses.append(Bus(bus_id=f"Bus{self.bus_counter}", current_stop=self.buses[0].current_stop, max_capacity=BUS_CAPACITY))
                print(f"[{hour:02d}시] {len(remaining_hourly)}명 남음 → {self.buses[-1].bus_id} 추가 투입")
                pairs2 = [(c.boarding_stop, c.getoff_stop) for c in remaining_hourly]
                stops2, _, _, _ = run_ga(pairs2, verbose=False)
                self._simulate_bus_run(self.buses[-1], stops2, hour_min, hour_end)

            # ----- 3) 시간대 종료 강제 정리 (한 시간 내 완전 처리) -----
            self._force_sweep_within_hour(hourly_customers, hour_min, hour_end)

            # ----- 4) 시간대별 리포트 -----
            served_this_hour = [
                c for c in self.customers
                if hasattr(c, "pickup_time") and hour_min <= c.pickup_time < hour_end
            ]
            if served_this_hour:
                print(f"\n[{hour:02d}시] 고객별 대기시간")
                for c in served_this_hour:
                    wait = c.pickup_time - c.time
                    h, m = divmod(max(0, wait), 60)
                    print(f" - Customer {c.customer_id}: 대기 {max(0, wait)}분 ({h:02d}:{m:02d}) @ {c.boarding_stop}→{c.getoff_stop}")
            else:
                print(f"\n[{hour:02d}시] 이 시간에 탑승한 고객 없음")

            hour_km = sum(b.total_distance - start_dist.get(b.bus_id, 0.0) for b in self.buses)
            print(f"[{hour:02d}시] 이동거리 합계: {hour_km:.2f} km")

            aband_delta = self.abandoned_customers - aband_start
            if aband_delta > 0:
                print(f"[{hour:02d}시] 포기 발생: {aband_delta}명")

            # 시간 경계 복귀(빈 시간대 + 다음 수요 적음)
            self._maybe_return_to_depot(hour, hour_end)

        # =========================
        # 최종 요약
        # =========================
        print("\n=== 시뮬레이션 총괄 ===")
        waits = [max(0, c.pickup_time - c.time) for c in self.customers if hasattr(c, "pickup_time")]
        if waits:
            avg = sum(waits) / len(waits)
            h, m = divmod(int(avg), 60)
            print(f"[평균 대기시간] {avg:.2f}분 ({h:02d}:{m:02d})  /  탑승 {len(waits)}명")
        total_km = sum(b.total_distance for b in self.buses)
        print(f"[총 이동거리] {total_km:.2f} km")
        print(f"[총 비용] {calculate_cost(total_km):,}원")
        print(f"[누적 포기] {self.abandoned_customers}명")
        return total_route_summary


# 실행
if __name__ == "__main__":
    sim = Simulation()
    sim.run()
