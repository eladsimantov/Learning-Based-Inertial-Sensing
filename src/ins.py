# src/ins.py
import pandas as pd
import numpy as np

class Sensor:
    """
    This is the base class for all sensors. 
    It contains the common attributes and methods.
    get_sensor_name() -> str: returns the name of the sensor
    set_calibration_data(**kwargs): sets the calibration data for the sensor
    get_calibration_data() -> dict: returns the calibration data for the sensor
    """
    def __init__(self, name: str):
        self.__name = name
        return
    
    def get_sensor_name(self) -> str:
        return self.__name
    
    def set_calibration_data(self, **kwargs):
        """
        This method sets the calibration data for the sensor. 
        Enter the calibration data as keyword arguments.
        """
        self._calibration_data = {**kwargs}
        return
    
    def get_calibration_data(self) -> dict:
        return self._calibration_data
    
    @staticmethod
    def calc_bias(mean_positive: np.ndarray, mean_negative: np.ndarray) -> np.ndarray:
        return (mean_positive + mean_negative) / 2
    
    @staticmethod
    def calc_scale_factor(mean_positive: np.ndarray, mean_negative: np.ndarray, forcing) -> np.ndarray:
        """ 
        For Gyroscope: forcing = omega
        For Accelerometer: forcing = g
        """
        return (mean_positive - mean_negative - 2 * forcing)/(2 * forcing)
        
            
class Gyroscope(Sensor):
    def __init__(self, name: str):
        super().__init__(name)
        return
        
    def get_M_errors_matrix(self) -> np.ndarray:
        w_dict = self.get_calibration_data()
        w = np.hstack((w_dict["wx_left"], w_dict["wx_right"], w_dict["wy_left"], w_dict["wy_right"], w_dict["wz_left"], w_dict["wz_right"]))
        M = self.calc_M_matrix(w)
        return M
    
    def get_bias(self) -> np.ndarray:
        w_dict = self.get_calibration_data()
        w_left = np.array([w_dict["wx_left"], w_dict["wy_left"], w_dict["wz_left"]])
        w_right = np.array([w_dict["wx_right"], w_dict["wy_right"], w_dict["wz_right"]])
        bias = self.calc_bias(w_left, w_right)
        return bias
    
    @staticmethod
    def calc_bias(mean_w_left, mean_w_right) -> np.ndarray:
        return np.array((mean_w_left + mean_w_right)/2)
    
    @staticmethod
    def calc_M_matrix(w: np.ndarray) -> np.ndarray:
        # The rational is: M @ A = z => M = z @ A^T @ (A @ A^T)^-1 = (Misalignment + ScaleFactor, bias) @(w;1)
        z = w
        A = np.array([
            [-1, 1, 0, 0, 0, 0], 
            [0, 0, -1, 1, 0, 0], 
            [0, 0, 0, 0, -1, 1], 
            [1, 1, 1, 1, 1, 1]])
        M = z @ A.T @ np.linalg.inv(A @ A.T)
        return M
    


class Accelerometer(Sensor):
    def __init__(self, name: str):
        super().__init__(name)
        return
    
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
