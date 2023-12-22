import time

class TimeRecord:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()

    def get_spend_time(self):
        elapsed_time = self.end_time - self.start_time
        return float('{:.5f}'.format(elapsed_time))

    