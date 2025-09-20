import pandas as pd
import numpy as np
import random
from collections import defaultdict
from route import get_distance_between

class Customer:
    def __init__(self, customer_id, boarding_stop, getoff_stop, time):
        self.customer_id = customer_id
        self.boarding_stop = boarding_stop
        self.getoff_stop = getoff_stop
        self.time = time  # 고객이 정류장에서 대기하기 시작하는 시간