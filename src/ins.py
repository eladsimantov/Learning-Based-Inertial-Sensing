# src/INS/INS.py

import pandas as pd
import numpy as np

class INS:
    def __init__(self, raw_data: pd.DataFrame, device_data: pd.DataFrame, time_data: pd.DataFrame):
        self._raw_data = raw_data
        self._device_data = device_data
        self._time_data = time_data
        return 
    
    def get_raw_data(self): 
        return self._raw_data

    def get_device_data(self): 
        return self._device_data

    def get_time_data(self): 
        return self._time_data   

    def calibrate_accelerometer(self):
        self.get_raw_data()
        return
    

    