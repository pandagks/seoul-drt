from parameters import load_fixed_customers
from customer import Customer
from utils import get_distance_between, calculate_cost
from bus import Bus
from ga_optimizer import run_ga
from statistics import mean, stdev
import heapq
import pandas as pd
import streamlit as st
import folium


class Simulation:
    def __init__(self):
        self.customers = []
        self.buses = []
        self.waiting_customers = {}
        self.current_time = 300  # 10:00부터 시작
        self.bus_counter = 1
        self.total_distance_across_runs = 0
        self.total_time_across_runs = 0
        self.fitness_all = []
        self.abandoned_customers = 0

    def generate_customers(self):
        fixed_customers = load_fixed_customers()
        for customer in fixed_customers:
            if customer.boarding_stop != customer.getoff_stop:
                self.customers.append(customer)
                self.waiting_customers.setdefault(customer.boarding_stop, []).append(customer)

    def run(self):
        self.generate_customers()
        self.buses.append(Bus(bus_id="Bus1", current_stop="00_오이도차고지", max_capacity=40))
        total_route_summury = {}
        for hour in range(5, 23):
            hour_min = hour * 60
            hourly_customers = [c for c in self.customers if hour_min <= c.time < hour_min + 60]
            hourly_ids = {c.customer_id for c in hourly_customers}
            remaining_customers = hourly_customers.copy()

            pairs = [(c.boarding_stop, c.getoff_stop) for c in hourly_customers]
            if not pairs:
                continue
            stops_to_visit, fitness_with_return, distance, minutes = run_ga(pairs, verbose=True)
            # 탑승/하차 인원 계산
            boarding_count = {}
            getoff_count = {}

            for c in hourly_customers:
                boarding_count[c.boarding_stop] = boarding_count.get(c.boarding_stop, 0) + 1
                getoff_count[c.getoff_stop] = getoff_count.get(c.getoff_stop, 0) + 1

            route_summary = []
            for stop in stops_to_visit:
                board = boarding_count.get(stop, 0)
                drop = getoff_count.get(stop, 0)
                label = stop
                info = []
                if board > 0:
                    info.append(f"{board}승차")
                if drop > 0:
                    info.append(f"{drop}하차")
                if info:
                    label += f"({', '.join(info)})"
                route_summary.append(label)

            print(f"[{hour}시 사이클] 방문 경로: {' → '.join(route_summary)}")
            total_route_summury[f'{hour}'] = route_summary
            self.total_distance_across_runs += distance
            self.total_time_across_runs += minutes
            self.fitness_all.extend(fitness_with_return)

            bus_index = 0
            while remaining_customers:
                if bus_index >= len(self.buses):
                    self.bus_counter += 1
                    new_bus = Bus(bus_id=f"Bus{self.bus_counter}", current_stop="00_오이도차고지", max_capacity=40)
                    self.buses.append(new_bus)

                bus = self.buses[bus_index]
                bus_index += 1

                unfinished = {c.customer_id for c in remaining_customers}
                while unfinished:
                    # --- FIX 2: 진행 상황 추적 플래그 초기화 ---
                    progress = False
                    # --- FIX 2 end ---

                    for i, stop in enumerate(stops_to_visit):
                        if i > 0:
                            prev = stops_to_visit[i - 1]
                            dist = get_distance_between(prev, stop)
                            # --- FIX 1: 이동시간 최소 1분 보장 + 진행 발생 표시 ---
                            if dist is None:
                                self.current_time += 1
                                progress = True
                            else:
                                bus.total_distance += dist
                                dt = max(1, int(dist * 3))
                                self.current_time += dt
                                progress = True
                            # --- FIX 1 end ---

                        bus.current_stop = stop

                        dropped = bus.drop_customer(stop, self.current_time)
                        for c in dropped:
                            c.dropoff_time = self.current_time
                            hour_, minute_ = divmod(self.current_time, 60)
                            print(f"[{hour_:02d}:{minute_:02d}] {c.customer_id}번 고객이 {bus.bus_id} 버스에서 하차 (정류장: {stop})")
                            # --- FIX 2: 하차 발생도 진행으로 간주 ---
                            progress = True
                            # --- FIX 2 end ---

                        waiting_list = list(self.waiting_customers.get(stop, []))
                        for c in waiting_list:
                            wait_time = self.current_time - c.time
                            if (
                                c.customer_id in hourly_ids and
                                c.time <= self.current_time and
                                wait_time <= 45 and
                                bus.can_board_customer()
                            ):
                                bus.board_customer(c, self.current_time)
                                self.waiting_customers[stop].remove(c)
                                c.pickup_time = self.current_time
                                hour_, minute_ = divmod(self.current_time, 60)
                                print(f"[{hour_:02d}:{minute_:02d}] {c.customer_id}번 고객이 {bus.bus_id} 버스에 탑승 (정류장: {stop}, 하차: {c.getoff_stop})")
                                # --- FIX 2: 탑승 발생도 진행으로 간주 ---
                                progress = True
                                # --- FIX 2 end ---
                            elif wait_time > 45:
                                self.waiting_customers[stop].remove(c)
                                self.abandoned_customers += 1
                                remaining_customers = [x for x in remaining_customers if x.customer_id != c.customer_id]
                                hour_, minute_ = divmod(self.current_time, 60)
                                print(f"[{hour_:02d}:{minute_:02d}] {c.customer_id}번 고객이 {stop}에서 대기 {wait_time}분 후 탑승 포기")
                                # (여기는 progress 굳이 True로 안 해도 시간은 흘러가므로 OK)

                    # --- FIX 2: 한 바퀴 돌았는데 아무 일도 없으면 1분 경과시켜 정체 탈출 ---
                    if not progress:
                        self.current_time += 1
                    # --- FIX 2 end ---

                    unfinished = {
                        c.customer_id
                        for c in remaining_customers
                        if c.customer_id not in {cust.customer_id for cust, _ in bus.finished_customers}
                    }

                bus.end_time = self.current_time
                remaining_customers = [
                    c for c in remaining_customers
                    if all(c.customer_id != cust.customer_id for cust, _ in bus.finished_customers)
                ]

            next_hour = (hour + 1) * 60 if hour < 22 else None
            if next_hour:
                time_gap = next_hour - self.current_time
                if time_gap >= 15:
                    for bus in self.buses:
                        if bus.current_stop != "00_오이도차고지":
                            dist = get_distance_between(bus.current_stop, "00_오이도차고지")
                            if dist:
                                bus.total_distance += dist
                                self.current_time += int(dist * 3)
                                bus.current_stop = "00_오이도차고지"
                                hour_, minute_ = divmod(self.current_time, 60)
                                print(f"[{hour_:02d}:{minute_:02d}] {bus.bus_id} 버스가 오이도차고지로 복귀하여 대기")
                else:
                    hour_, minute_ = divmod(self.current_time, 60)
                    print(f"[{hour_:02d}:{minute_:02d}] 버스들이 대기 없이 다음 정류장 이동 예정")

        print("\n=== 시뮬레이션 종료 ===")

        print("[GA 최종 요약]")
        print("\n[고객별 대기 시간 요약]")
        total_waiting = 0
        count = 0
        for customer in self.customers:
            if hasattr(customer, 'pickup_time'):
                wait = customer.pickup_time - customer.time
                total_waiting += wait
                count += 1
                h, m = divmod(wait, 60)
                print(f"Customer {customer.customer_id}: 대기 {wait}분 ({h:02d}:{m:02d})")
        if count:
            avg = total_waiting / count
            h, m = divmod(int(avg), 60)
            print(f"\n[평균 대기 시간] {avg:.2f}분 ({h:02d}:{m:02d}) - 총 {count}명")
        print(f"\n[탑승 포기 고객 수] {self.abandoned_customers}명")
        print(f"총 누적 거리: {self.total_distance_across_runs:.2f} km")
        print(f"총 누적 시간: {self.total_time_across_runs}분")
        print(f"총 예상 비용: {calculate_cost(self.total_distance_across_runs):,}원")
        return total_route_summury


if __name__ == "__main__":
    sim = Simulation()
    total_route = sim.run()
    # print(total_route.keys())
