# src/ins.py
import pandas as pd
import numpy as np

class Accelerometer:
    def __init__(self, name: str):
        self.__name = name
        return 
    
    def set_calibration_data(self, 
                             fx_down: np.ndarray, fx_up : np.ndarray, 
                             fy_down: np.ndarray, fy_up: np.ndarray, 
                             fz_down: np.ndarray, fz_up: np.ndarray, 
                             gravity: float):
        self._fx_down = fx_down
        self._fx_up = fx_up
        self._fy_down = fy_down
        self._fy_up = fy_up
        self._fz_down = fz_down
        self._fz_up = fz_up
        self._gravity = gravity
        return

    def get_calibration_data(self) -> dict:
        return {
            'fx_up': self._fx_up,
            'fx_down': self._fx_down,
            'fy_up': self._fy_up,
            'fy_down': self._fy_down,
            'fz_up': self._fz_up,
            'fz_down': self._fz_down,
            'g': self._gravity
        }
    
    def get_M_errors_matrix(self) -> np.ndarray:
        f_dict = self.get_calibration_data()
        f = np.hstack((f_dict["fx_down"], f_dict["fx_up"], f_dict["fy_down"], f_dict["fy_up"], f_dict["fz_down"], f_dict["fz_up"]))
        M = self.calc_M_matrix(f, f_dict['g'])
        return M 

    @staticmethod
    def calc_bias(mean_f_up, mean_f_down) -> np.ndarray:
        return np.array((mean_f_up + mean_f_down)/2)
    
    @staticmethod
    def calc_scale_factor(mean_f_up, mean_f_down, gravity) -> np.ndarray:
        return  np.array((mean_f_up - mean_f_down - 2 * gravity)/(2 * gravity))

    @staticmethod
    def calc_M_matrix(f: np.ndarray, g: float) -> np.ndarray:
        # The rational is: M @ A = z => M = z @ A^T @ (A @ A^T)^-1 = (Misalignment + ScaleFactor, bias) @(f;1)
        
        G_mat = np.array([[-g, g, 0, 0, 0, 0], 
                          [0, 0, -g, g, 0, 0], 
                          [0, 0, 0, 0, -g, g]])
        z = f - G_mat
        A = np.array([
            [-g, g,  0, 0,  0, 0], 
            [0,  0, -g, g,  0, 0], 
            [0,  0,  0, 0, -g, g], 
            [1,  1,  1, 1,  1, 1]])
        M = z @ A.T @ np.linalg.inv(A @ A.T)
        return M
