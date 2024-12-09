# src/INS/INS.py

import pandas as pd
import numpy as np

class Accelerometer:
    def __init__(self):
        return 
    
    def set_calibration_data(self, fx_up, fx_down, fy_up, fy_down, fz_up, fz_down):
        """
        Enter data as a dictionary of dataframes called: 'raw_data', 'device_data', 'time_data'
        """
        self._fx_up = fx_up
        self._fx_down = fx_down
        self._fy_up = fy_up
        self._fy_down = fy_down
        self._fz_up = fz_up
        self._fz_down = fz_down
        return

    def get_calibration_data(self):
        return {
            'fx_up': self._fx_up,
            'fx_down': self._fx_down,
            'fy_up': self._fy_up,
            'fy_down': self._fy_down,
            'fz_up': self._fz_up,
            'fz_down': self._fz_down
        }

    @staticmethod
    def calc_bias(mean_f_up, mean_f_down):
        return (mean_f_up + mean_f_down)/2
    
    @staticmethod
    def calc_scale_factor(mean_f_up, mean_f_down, gravity):
        return  (mean_f_up - mean_f_down - 2 * gravity)/(2 * gravity)

        

    