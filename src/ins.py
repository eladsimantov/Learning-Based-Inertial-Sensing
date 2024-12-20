# src/ins.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

class Sensor:
    """
    This is the base class for all sensors. 
    It contains the common attributes and methods.
    get_sensor_name() -> str: returns the name of the sensor
    set_calibration_data(**kwargs): sets the calibration data for the sensor
    get_calibration_data() -> dict: returns the calibration data for the sensor
    calc_bias(mean_positive: np.ndarray, mean_negative: np.ndarray) -> np.ndarray: 
        returns the bias of the sensor assuming no misalignment
    calc_scale_factor(mean_positive: np.ndarray, mean_negative: np.ndarray, forcing) -> np.ndarray:
        returns the scale factor of the sensor assuming no misalignment
    """
    def __init__(self, name: str, **kwargs):
        """
        name: str: name of the sensor
        bias: np.ndarray: bias of the sensor as a 3x1 vector
        scale_factor: np.ndarray: scale factor of the sensor as a diagonal 3x3 matrix
        M_errors_matrix: np.ndarray: M matrix of the sensor as a 3x6 matrix
        """
        self.__name = name
        self._calibration_data = {}
        self._bias = kwargs["bias"] if "bias" in kwargs else None
        self._scale_factor = kwargs["scale_factor"] if "scale_factor" in kwargs else None
        self._M_errors_matrix = kwargs["M_errors_matrix"] if "M_errors_matrix" in kwargs else None
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
    
    @staticmethod 
    def plot_time_series(x_data: np.ndarray, y_data: np.ndarray, title: str, x_label: str):
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        axs[0].plot(x_data, y_data[:, 0], label='X-axis', color='r')
        axs[0].set_title(f'{title} - X-axis')
        axs[0].set_ylabel('X-axis')
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].plot(x_data, y_data[:, 1], label='Y-axis', color='g')
        axs[1].set_title(f'{title} - Y-axis')
        axs[1].set_ylabel('Y-axis')
        axs[1].legend()
        axs[1].grid(True)
        
        axs[2].plot(x_data, y_data[:, 2], label='Z-axis', color='b')
        axs[2].set_title(f'{title} - Z-axis')
        axs[2].set_xlabel(x_label)
        axs[2].set_ylabel('Z-axis')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        return
        
            
class Gyroscope(Sensor):
    def __init__(self, name: str):
        super().__init__(name)
        return
        
    # def compute_M_errors_matrix(self) -> np.ndarray:
    #     if self.get_calibration_data() == {}:
    #         raise ValueError("Calibration data is not set\n Use set_calibration_data() method to set the calibration data")
    #     w_dict = self.get_calibration_data() 
    #     w = np.hstack((w_dict["wx_left"], w_dict["wx_right"], w_dict["wy_left"], w_dict["wy_right"], w_dict["wz_left"], w_dict["wz_right"]))
    #     M = self.calc_M_matrix(w)

    #     # Save the M matrix to object if it is not already saved
    #     if self._M_errors_matrix is None:
    #         self._M_errors_matrix = M
    #     return M
    
    def compute_bias(self) -> np.ndarray:
        w_dict = self.get_calibration_data()
        w_left = np.array([w_dict["wx_left"][0], w_dict["wy_left"][1], w_dict["wz_left"][2]])
        w_right = np.array([w_dict["wx_right"][0], w_dict["wy_right"][1], w_dict["wz_right"][2]])
        bias = self.calc_bias(w_left, w_right)
        return bias
    
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
    
    def compute_M_errors_matrix(self) -> np.ndarray:
        if self.get_calibration_data() == {}:
            raise ValueError("Calibration data is not set\n Use set_calibration_data() method to set the calibration data")
        f_dict = self.get_calibration_data()
        f = np.hstack((f_dict["fx_down"], f_dict["fx_up"], f_dict["fy_down"], f_dict["fy_up"], f_dict["fz_down"], f_dict["fz_up"]))
        M = self.calc_M_matrix(f, f_dict['g'])
        # Save the M matrix to object if it is not already saved
        if self._M_errors_matrix is None:
            self._M_errors_matrix = M
        return M 

    def compute_bias(self) -> np.ndarray:
        f_dict = self.get_calibration_data()
        f_down = np.array([f_dict["fx_down"][0], f_dict["fy_down"][1], f_dict["fz_down"][2]])
        f_up = np.array([f_dict["fx_up"][0], f_dict["fy_up"][1], f_dict["fz_up"][2]])
        bias = self.calc_bias(f_down, f_up)
        return bias
    
    def compute_scale_factor(self) -> np.ndarray:
        f_dict = self.get_calibration_data()
        f_down = np.array([f_dict["fx_down"][0], f_dict["fy_down"][1], f_dict["fz_down"][2]])
        f_up = np.array([f_dict["fx_up"][0], f_dict["fy_up"][1], f_dict["fz_up"][2]])
        scale_factor = self.calc_scale_factor(f_up, f_down, f_dict['g'])
        return np.diag([scale_factor[0][0], scale_factor[1][0], scale_factor[2][0]])
    
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
