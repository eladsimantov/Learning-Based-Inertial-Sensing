# HW 1
# The homework assignments is on calibrating the Inertial sensor in our cell phones via the 6-position method. 
# The first part will be to use recorded data to calibrate the accelerometer, the second part will be to calibrate the gyroscopes and the third part will be to analyze recorded data of "shaking the smartphone". Finally, we will compare different phones with one another and with our expectations from their specifications. 

# In[1]:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
np.set_printoptions(precision=4, suppress=True)

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
        plt.pause(0.1)
        return
        
            
class Gyroscope(Sensor):
    def __init__(self, name: str):
        super().__init__(name)
        return


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

# Load the data in new format
def load_data(person):
    for folder in os.listdir(os.path.join("..", "data", person)):
        df_name = f"{person.lower()}_{folder.lower().replace(' ', '_')}"
        if "Raw Data.csv" in os.listdir(os.path.join("..", "data", person, folder)):
            globals()[df_name] = pd.read_csv(os.path.join("..", "data", person, folder, "Raw Data.csv"))
            print(f"Loaded {df_name} in {folder}")
        elif "Accelerometer.csv" in os.listdir(os.path.join("..", "data", person, folder)):
            # This case means there is also gyroscope data
            globals()[df_name+"_accelerometer"] = pd.read_csv(os.path.join("..", "data", person, folder, "Accelerometer.csv"))
            print(f"Loaded {df_name}_accelerometer in {folder}")
            globals()[df_name+"_gyroscope"] = pd.read_csv(os.path.join("..", "data", person, folder, "Gyroscope.csv"))
            print(f"Loaded {df_name}_gyroscope in {folder}")
        else:
            print(f"Could not find data for {person} in {folder}")


def remove_rows_by_time(df, start_time, end_time):
    return df[(df["Time (s)"] >= start_time) & (df["Time (s)"] <= end_time)]

def get_mean_accelerations(df):
    return df[["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)"]].mean().to_numpy().reshape([3,1])

def get_mean_gyroscopes(df):
    return df[["Gyroscope x (rad/s)", "Gyroscope y (rad/s)", "Gyroscope z (rad/s)"]].mean().to_numpy().reshape([3,1])



# ## Elad

# ### Accelerometers

# In[2]:


# Load the data in old format
path_accel_300sec_face_up = os.path.join("..", "data", "Elad", "old_format", "Acceleration_300sec_FaceUp")
path_accel_300sec_face_down = os.path.join("..", "data", "Elad", "old_format", "Acceleration_300sec_FaceDown")
path_accel_300sec_side_positive = os.path.join("..", "data", "Elad", "old_format", "Acceleration_300sec_SidePositive")
path_accel_300sec_side_negative = os.path.join("..", "data", "Elad", "old_format", "Acceleration_300sec_SideNegative")
path_accel_300sec_vertical_front = os.path.join("..", "data", "Elad", "old_format", "Acceleration_300sec_VerticalFront")
path_accel_300sec_vertical_back = os.path.join("..", "data", "Elad", "old_format", "Acceleration_300sec_VerticalBack")

elad_acceleration_with_g_300sec_up = pd.read_csv(os.path.join(path_accel_300sec_face_up, "Raw Data.csv")) # This had a delimiter of ","
elad_acceleration_with_g_300sec_down = pd.read_csv(os.path.join(path_accel_300sec_face_down, "Raw Data.csv"), sep="\t", engine="python")
elad_acceleration_with_g_300sec_left = pd.read_csv(os.path.join(path_accel_300sec_side_positive, "Raw Data.csv"), sep="\t", engine="python")
elad_acceleration_with_g_300sec_right = pd.read_csv(os.path.join(path_accel_300sec_side_negative, "Raw Data.csv"), sep="\t", engine="python")
elad_acceleration_with_g_300sec_front = pd.read_csv(os.path.join(path_accel_300sec_vertical_front, "Raw Data.csv"), sep="\t", engine="python")
elad_acceleration_with_g_300sec_back = pd.read_csv(os.path.join(path_accel_300sec_vertical_back, "Raw Data.csv"), sep="\t", engine="python")

# Load data in new format
load_data("Elad")


# In[3]:


# 5sec experiment
START_TIME_BUFFER = 0.1 # seconds to remove from the start
fz_up_5sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_5sec_up, START_TIME_BUFFER, 5 + START_TIME_BUFFER))
fz_down_5sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_5sec_down, START_TIME_BUFFER, 5 + START_TIME_BUFFER))
fx_up_5sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_5sec_left, START_TIME_BUFFER, 5 + START_TIME_BUFFER))
fx_down_5sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_5sec_right, START_TIME_BUFFER, 5 + START_TIME_BUFFER))
fy_up_5sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_5sec_front, START_TIME_BUFFER, 5 + START_TIME_BUFFER))
fy_down_5sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_5sec_back, START_TIME_BUFFER, 5 + START_TIME_BUFFER))

# 60sec experiment
fz_up_60sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_60sec_up, START_TIME_BUFFER, 60 + START_TIME_BUFFER))
fz_down_60sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_60sec_down, START_TIME_BUFFER, 60 + START_TIME_BUFFER))
fx_up_60sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_60sec_left, START_TIME_BUFFER, 60 + START_TIME_BUFFER))
fx_down_60sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_60sec_right, START_TIME_BUFFER, 60 + START_TIME_BUFFER))
fy_up_60sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_60sec_front, START_TIME_BUFFER, 60 + START_TIME_BUFFER))
fy_down_60sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_60sec_back, START_TIME_BUFFER, 60 + START_TIME_BUFFER))

# 300sec experiment
fz_up_300sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_300sec_up, START_TIME_BUFFER, 300 + START_TIME_BUFFER))
fz_down_300sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_300sec_down, START_TIME_BUFFER, 300 + START_TIME_BUFFER))
fx_up_300sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_300sec_left, START_TIME_BUFFER, 300 + START_TIME_BUFFER))
fx_down_300sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_300sec_right, START_TIME_BUFFER, 300 + START_TIME_BUFFER))
fy_up_300sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_300sec_front, START_TIME_BUFFER, 300 + START_TIME_BUFFER))
fy_down_300sec = get_mean_accelerations(remove_rows_by_time(elad_acceleration_with_g_300sec_back, START_TIME_BUFFER, 300 + START_TIME_BUFFER))


# In[4]:


g_true = 9.807 # m/s^2

# create sensors objects for the accelerometers for the different time periods (5, 60, 300 seconds)
Elad_acc_300 = Accelerometer(name="Elad accelerometer 300 sec")
Elad_acc_300.set_calibration_data(
    fx_up=fx_up_300sec, fx_down=fx_down_300sec, 
    fy_up=fy_up_300sec, fy_down=fy_down_300sec, 
    fz_up=fz_up_300sec, fz_down=fz_down_300sec, 
    g=g_true)

Elad_acc_60 = Accelerometer(name="Elad accelerometer 60 sec")
Elad_acc_60.set_calibration_data(
    fx_up=fx_up_60sec, fx_down=fx_down_60sec, 
    fy_up=fy_up_60sec, fy_down=fy_down_60sec, 
    fz_up=fz_up_60sec, fz_down=fz_down_60sec, 
    g=g_true)

Elad_acc_5 = Accelerometer(name="Elad accelerometer 5 sec")
Elad_acc_5.set_calibration_data(
    fx_up=fx_up_5sec, fx_down=fx_down_5sec, 
    fy_up=fy_up_5sec, fy_down=fy_down_5sec, 
    fz_up=fz_up_5sec, fz_down=fz_down_5sec, 
    g=g_true)



# In[5]:


# Compute bias and scale factor for each accelerometer
bias_300 = Elad_acc_300.compute_bias()
scale_factor_300 = Elad_acc_300.compute_scale_factor()

bias_60 = Elad_acc_60.compute_bias()
scale_factor_60 = Elad_acc_60.compute_scale_factor()

bias_5 = Elad_acc_5.compute_bias()
scale_factor_5 = Elad_acc_5.compute_scale_factor()

# Print stats for each accelerometer
print(f"Stats for {Elad_acc_300.get_sensor_name()} sensors:")
print(f"bias x: {bias_300[0][0]}\nbias y: {bias_300[1][0]}\nbias z: {bias_300[2][0]}")
print(f"scale factor x: {scale_factor_300[0,0]}\nscale factor y: {scale_factor_300[1,1]}\nscale factor z: {scale_factor_300[2,2]}\n")

print(f"Stats for {Elad_acc_60.get_sensor_name()} sensors:")
print(f"bias x: {bias_60[0][0]}\nbias y: {bias_60[1][0]}\nbias z: {bias_60[2][0]}")
print(f"scale factor x: {scale_factor_60[0,0]}\nscale factor y: {scale_factor_60[1,1]}\nscale factor z: {scale_factor_60[2,2]}\n")

print(f"Stats for {Elad_acc_5.get_sensor_name()} sensors:")
print(f"bias x: {bias_5[0][0]}\nbias y: {bias_5[1][0]}\nbias z: {bias_5[2][0]}")
print(f"scale factor x: {scale_factor_5[0,0]}\nscale factor y: {scale_factor_5[1,1]}\nscale factor z: {scale_factor_5[2,2]}")


# #### Q2.5 - Calculate Scale Factor, Bias and Misalignment using LSM

# In[6]:


# Calculate the errors matrix M = [SF + M | b]
M300 = Elad_acc_300.compute_M_errors_matrix()
M60 = Elad_acc_60.compute_M_errors_matrix()
M5 = Elad_acc_5.compute_M_errors_matrix()


# In[7]:


print(f"Stats for {Elad_acc_300.get_sensor_name()} sensors:")
print(f"M Matrix:\n{M300}\n")

print(f"Stats for {Elad_acc_60.get_sensor_name()} sensors:")
print(f"M Matrix:\n{M60}\n")

print(f"Stats for {Elad_acc_5.get_sensor_name()} sensors:")
print(f"M Matrix:\n{M5}")


# #### Q2.6 - position error in z axis

# In[8]:


position_error = lambda time, bias : (0.5*bias*time**2)

# according to the no misalignment assumption
print("No misalignment case position error:")
print(f"Position error Z after 300sec: {position_error(300, bias_300[2][0])}")
print(f"Position error Z after 60sec: {position_error(60, bias_60[2][0])}")
print(f"Position error Z after 5sec: {position_error(5, bias_5[2][0])}\n")

# according the misalignment case
print("Misalignment case position error:")
print(f"Position error Z after 300sec: {position_error(300, M300[2,3])}")
print(f"Position error Z after 60sec: {position_error(60, M60[2,3])}")
print(f"Position error Z after 5sec: {position_error(5, M5[2,3])}")



# #### Q2.7 - calculate the bias based on average of all three measurements (5 + 60 + 300)
# 

# In[9]:


Elad_acc_avg = Accelerometer(name="Elad accelerometer average")
fx_up_avg = (fx_up_300sec + fx_up_60sec + fx_up_5sec) / 3
fx_down_avg = (fx_down_300sec + fx_down_60sec + fx_down_5sec) / 3
fy_up_avg = (fy_up_300sec + fy_up_60sec + fy_up_5sec) / 3
fy_down_avg = (fy_down_300sec + fy_down_60sec + fy_down_5sec) / 3
fz_up_avg = (fz_up_300sec + fz_up_60sec + fz_up_5sec) / 3
fz_down_avg = (fz_down_300sec + fz_down_60sec + fz_down_5sec) / 3

Elad_acc_avg.set_calibration_data(
    fx_up=fx_up_avg, fx_down=fx_down_avg, 
    fy_up=fy_up_avg, fy_down=fy_down_avg, 
    fz_up=fz_up_avg, fz_down=fz_down_avg, 
    g=g_true)

bias_avg = Elad_acc_avg.compute_bias()

print(f"Stats for {Elad_acc_avg.get_sensor_name()} sensors:")
print(f"bias x: {bias_avg[0][0]}\nbias y: {bias_avg[1][0]}\nbias z: {bias_avg[2][0]}")


# ### Gyroscopes

# #### Q3.1

# In[10]:


omega_earth = 7.292115*(10**-5) # m/s^2
latitude_in_haifa = 32.794046 # degrees
omega_true = omega_earth*np.cos(90-latitude_in_haifa)

# 5sec experiment
wz_up_5sec = get_mean_gyroscopes(remove_rows_by_time(elad_gyroscope_rotation_rate_5sec_up, START_TIME_BUFFER, 5 + START_TIME_BUFFER))
wz_down_5sec = get_mean_gyroscopes(remove_rows_by_time(elad_gyroscope_rotation_rate_5sec_down, START_TIME_BUFFER, 5 + START_TIME_BUFFER))

# 60sec experiment
wz_up_60sec = get_mean_gyroscopes(remove_rows_by_time(elad_gyroscope_rotation_rate_60sec_up, START_TIME_BUFFER, 60 + START_TIME_BUFFER))
wz_down_60sec = get_mean_gyroscopes(remove_rows_by_time(elad_gyroscope_rotation_rate_60sec_down, START_TIME_BUFFER, 60 + START_TIME_BUFFER))

# create sensors objects for the gyros for the different time periods (5, 60 seconds)
Elad_gyr_60 = Gyroscope(name="Elad gyroscope 60 sec")
Elad_gyr_5 = Gyroscope(name="Elad gyroscope 5 sec")

bias_60 = Elad_gyr_60.calc_bias(wz_up_60sec, wz_down_60sec)
scale_factor_60 = Elad_gyr_60.calc_scale_factor(wz_up_60sec, wz_down_60sec, omega_true)
bias_5 = Elad_gyr_5.calc_bias(wz_up_5sec, wz_down_5sec)
scale_factor_5 = Elad_gyr_5.calc_scale_factor(wz_up_5sec, wz_down_5sec, omega_true)

# Print stats for each gyroscope
print(f"Stats for {Elad_gyr_60.get_sensor_name()} sensors:")
print(f"bias z: {bias_60[2][0]}")
print(f"scale factor z: {scale_factor_60[2][0]}\n")

print(f"Stats for {Elad_gyr_5.get_sensor_name()} sensors:")
print(f"bias z: {bias_5[2][0]}")
print(f"scale factor z: {scale_factor_5[2][0]}")


# #### Q3.2

# In[11]:


elad_partA_32_scenario = remove_rows_by_time(elad_gyroscope_rotation_rate_32_scenario, 0, 60)
elad_partB_32_scenario = remove_rows_by_time(elad_gyroscope_rotation_rate_32_scenario, 70, 130)
Elad_gyr_32_scenario = Gyroscope(name="Elad gyroscope 32 sec scenario")
bias = Elad_gyr_32_scenario.calc_bias(get_mean_gyroscopes(elad_partA_32_scenario), get_mean_gyroscopes(elad_partB_32_scenario))
scale_factor = Elad_gyr_32_scenario.calc_scale_factor(get_mean_gyroscopes(elad_partA_32_scenario), get_mean_gyroscopes(elad_partB_32_scenario), omega_true)

print("{}\nBias: \n{}\nScale Factor \n{}".format(Elad_gyr_32_scenario.get_sensor_name(), bias[2][0], scale_factor[2][0]))


# #### Q3.3

# In[12]:


# take average of measurements
Elad_gyr_average = Gyroscope(name="Elad gyroscope average")
bias_average = Elad_gyr_average.calc_bias((wz_up_60sec + wz_up_5sec)/2, (wz_down_60sec + wz_down_5sec)/2)
scale_factor_average = Elad_gyr_average.calc_scale_factor((wz_up_60sec + wz_up_5sec)/2, (wz_down_60sec + wz_down_5sec)/2, omega_true)

print(f"Stats for {Elad_gyr_average.get_sensor_name()} sensors:")
print(f"bias z: {bias_average[2][0]}")
print(f"scale factor z: {scale_factor_average[2][0]}")


# ### Accelerometers and Gyroscopes

# #### Q4.2

# In[13]:


# Load experiment data
# Get the data for the accelerometer
f_up_300 = get_mean_accelerations(remove_rows_by_time(elad_accelerometer_and_gyroscope_300sec_accelerometer, START_TIME_BUFFER, 300 + START_TIME_BUFFER))
f_up_60 = get_mean_accelerations(remove_rows_by_time(elad_accelerometer_and_gyroscope_60sec_accelerometer, START_TIME_BUFFER, 60 + START_TIME_BUFFER))
f_up_5 = get_mean_accelerations(remove_rows_by_time(elad_accelerometer_and_gyroscope_5sec_accelerometer, START_TIME_BUFFER, 5 + START_TIME_BUFFER))

# Get the data for the gyroscopes
omega_up_300 = get_mean_gyroscopes(remove_rows_by_time(elad_accelerometer_and_gyroscope_300sec_gyroscope, START_TIME_BUFFER, 300 + START_TIME_BUFFER))
omega_up_60 = get_mean_gyroscopes(remove_rows_by_time(elad_accelerometer_and_gyroscope_60sec_gyroscope, START_TIME_BUFFER, 60 + START_TIME_BUFFER))
omega_up_5 = get_mean_gyroscopes(remove_rows_by_time(elad_accelerometer_and_gyroscope_5sec_gyroscope, START_TIME_BUFFER, 5 + START_TIME_BUFFER))

f_bias_300 = f_up_300 - np.array([[0],[0], [g_true]])
f_bias_60 = f_up_60 - np.array([[0],[0], [g_true]])
f_bias_5 = f_up_5 - np.array([[0],[0], [g_true]])

omega_bias_300 = omega_up_300 - np.array([[0],[0], [omega_true]])
omega_bias_60 = omega_up_60 - np.array([[0],[0], [omega_true]])
omega_bias_5 = omega_up_5 - np.array([[0],[0], [omega_true]])

# print(f"Acceleration bias Z 300 sec: {f_bias_300[2][0]}")
# print(f"Acceleration bias Z 60 sec: {f_bias_60[2][0]}")
# print(f"Acceleration bias Z 5 sec: {f_bias_5[2][0]}")
# print(f"Gyroscope bias Z 300 sec: {omega_bias_300[2][0]}")
# print(f"Gyroscope bias Z 60 sec: {omega_bias_60[2][0]}")
# print(f"Gyroscope bias Z 5 sec: {omega_bias_5[2][0]}")

print(f"Stats for 300sec:")
print(f"accel bias x: {f_bias_300[0][0]}\naccel bias y: {f_bias_300[1][0]}\naccel bias z: {f_bias_300[2][0]}")
print(f"gyro bias x: {omega_bias_300[0][0]}\ngyro bias y: {omega_bias_300[1][0]}\ngyro bias z: {omega_bias_300[2][0]}")
print(f"Stats for 60sec:")
print(f"accel bias x: {f_bias_60[0][0]}\naccel bias y: {f_bias_60[1][0]}\naccel bias z: {f_bias_60[2][0]}")
print(f"gyro bias x: {omega_bias_60[0][0]}\ngyro bias y: {omega_bias_60[1][0]}\ngyro bias z: {omega_bias_60[2][0]}")
print(f"Stats for 5sec:")
print(f"accel bias x: {f_bias_5[0][0]}\naccel bias y: {f_bias_5[1][0]}\naccel bias z: {f_bias_5[2][0]}")
print(f"gyro bias x: {omega_bias_5[0][0]}\ngyro bias y: {omega_bias_5[1][0]}\ngyro bias z: {omega_bias_5[2][0]}")


# #### Q4.3

# In[14]:


Elad_shake_accel = remove_rows_by_time(elad_accelerometer_and_gyroscope_30sec_shake_accelerometer, START_TIME_BUFFER, 30 + START_TIME_BUFFER)
Elad_shake_gyro = remove_rows_by_time(elad_accelerometer_and_gyroscope_30sec_shake_gyroscope, START_TIME_BUFFER, 30 + START_TIME_BUFFER)

Elad_shake = Sensor(name="Elad shake accelerometer")
Elad_shake.plot_time_series(Elad_shake_accel["Time (s)"].to_numpy(), Elad_shake_accel[["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)"]].to_numpy(), "Accelerations (m/s^2)", "Time (s)")

# #### Q4.4

# In[15]:


# Load experiment data
# Get the data for the accelerometer
f_up_300 = get_mean_accelerations(remove_rows_by_time(elad_accelerometer_and_gyroscope_300sec_aftershake_accelerometer, START_TIME_BUFFER, 300 + START_TIME_BUFFER))
f_up_60 = get_mean_accelerations(remove_rows_by_time(elad_accelerometer_and_gyroscope_60sec_aftershake_accelerometer, START_TIME_BUFFER, 60 + START_TIME_BUFFER))
f_up_5 = get_mean_accelerations(remove_rows_by_time(elad_accelerometer_and_gyroscope_5sec_aftershake_accelerometer, START_TIME_BUFFER, 5 + START_TIME_BUFFER))

# Get the data for the gyroscopes
omega_up_300 = get_mean_gyroscopes(remove_rows_by_time(elad_accelerometer_and_gyroscope_300sec_aftershake_gyroscope, START_TIME_BUFFER, 300 + START_TIME_BUFFER))
omega_up_60 = get_mean_gyroscopes(remove_rows_by_time(elad_accelerometer_and_gyroscope_60sec_aftershake_gyroscope, START_TIME_BUFFER, 60 + START_TIME_BUFFER))
omega_up_5 = get_mean_gyroscopes(remove_rows_by_time(elad_accelerometer_and_gyroscope_5sec_aftershake_gyroscope, START_TIME_BUFFER, 5 + START_TIME_BUFFER))

f_bias_300 = f_up_300 - np.array([[0],[0], [g_true]])
f_bias_60 = f_up_60 - np.array([[0],[0], [g_true]])
f_bias_5 = f_up_5 - np.array([[0],[0], [g_true]])

omega_bias_300 = omega_up_300 - np.array([[0],[0], [omega_true]])
omega_bias_60 = omega_up_60 - np.array([[0],[0], [omega_true]])
omega_bias_5 = omega_up_5 - np.array([[0],[0], [omega_true]])

print(f"Stats for 300sec:")
print(f"accel bias x: {f_bias_300[0][0]}\naccel bias y: {f_bias_300[1][0]}\naccel bias z: {f_bias_300[2][0]}")
print(f"gyro bias x: {omega_bias_300[0][0]}\ngyro bias y: {omega_bias_300[1][0]}\ngyro bias z: {omega_bias_300[2][0]}")
print(f"Stats for 60sec:")
print(f"accel bias x: {f_bias_60[0][0]}\naccel bias y: {f_bias_60[1][0]}\naccel bias z: {f_bias_60[2][0]}")
print(f"gyro bias x: {omega_bias_60[0][0]}\ngyro bias y: {omega_bias_60[1][0]}\ngyro bias z: {omega_bias_60[2][0]}")
print(f"Stats for 5sec:")
print(f"accel bias x: {f_bias_5[0][0]}\naccel bias y: {f_bias_5[1][0]}\naccel bias z: {f_bias_5[2][0]}")
print(f"gyro bias x: {omega_bias_5[0][0]}\ngyro bias y: {omega_bias_5[1][0]}\ngyro bias z: {omega_bias_5[2][0]}")


# ## Ben

# In[16]:


load_data("Ben")


# ### Accelerometers

# In[17]:


# 5sec experiment
fz_up_5sec = get_mean_accelerations(ben_acceleration_with_g_5)
fz_down_5sec = get_mean_accelerations(ben_acceleration_with_g_5_flip)
fx_up_5sec = get_mean_accelerations(ben_acceleration_with_g_left_5)
fx_down_5sec = get_mean_accelerations(ben_acceleration_with_g_right_5)
fy_up_5sec = get_mean_accelerations(ben_acceleration_with_g_stand_5)
fy_down_5sec = get_mean_accelerations(ben_acceleration_with_g_stand_5_flip)

# 60sec experiment
fz_up_60sec = get_mean_accelerations(ben_acceleration_with_g_60)
fz_down_60sec = get_mean_accelerations(ben_acceleration_with_g_60_flip)
fx_up_60sec = get_mean_accelerations(ben_acceleration_with_g_left_60)
fx_down_60sec = get_mean_accelerations(ben_acceleration_with_g_right_60)
fy_up_60sec = get_mean_accelerations(ben_acceleration_with_g_stand_60)
fy_down_60sec = get_mean_accelerations(ben_acceleration_with_g_stand_60_flip)

# 300sec experiment
fz_up_300sec = get_mean_accelerations(ben_acceleration_with_g_300)
fz_down_300sec = get_mean_accelerations(ben_acceleration_with_g_300_flip)
fx_up_300sec = get_mean_accelerations(ben_acceleration_with_g_left_300)
fx_down_300sec = get_mean_accelerations(ben_acceleration_with_g_right_300)
fy_up_300sec = get_mean_accelerations(ben_acceleration_with_g_stand_300)
fy_down_300sec = get_mean_accelerations(ben_acceleration_with_g_stand_300_flip)


# In[18]:


g_true = 9.807 # m/s^2

# create sensors objects for the accelerometers for the different time periods (5, 60, 300 seconds)
Ben_acc_300 = Accelerometer(name="Ben accelerometer 300 sec")
Ben_acc_300.set_calibration_data(
    fx_up=fx_up_300sec, fx_down=fx_down_300sec, 
    fy_up=fy_up_300sec, fy_down=fy_down_300sec, 
    fz_up=fz_up_300sec, fz_down=fz_down_300sec, 
    g=g_true)

Ben_acc_60 = Accelerometer(name="Ben accelerometer 60 sec")
Ben_acc_60.set_calibration_data(
    fx_up=fx_up_60sec, fx_down=fx_down_60sec, 
    fy_up=fy_up_60sec, fy_down=fy_down_60sec, 
    fz_up=fz_up_60sec, fz_down=fz_down_60sec, 
    g=g_true)

Ben_acc_5 = Accelerometer(name="Ben accelerometer 5 sec")
Ben_acc_5.set_calibration_data(
    fx_up=fx_up_5sec, fx_down=fx_down_5sec, 
    fy_up=fy_up_5sec, fy_down=fy_down_5sec, 
    fz_up=fz_up_5sec, fz_down=fz_down_5sec, 
    g=g_true)


# In[19]:


# Compute bias and scale factor for each accelerometer
bias_300 = Ben_acc_300.compute_bias()
scale_factor_300 = Ben_acc_300.compute_scale_factor()

bias_60 = Ben_acc_60.compute_bias()
scale_factor_60 = Ben_acc_60.compute_scale_factor()

bias_5 = Ben_acc_5.compute_bias()
scale_factor_5 = Ben_acc_5.compute_scale_factor()

# Print stats for each accelerometer
print(f"Stats for {Ben_acc_300.get_sensor_name()} sensors:")
print(f"bias x: {bias_300[0][0]}\nbias y: {bias_300[1][0]}\nbias z: {bias_300[2][0]}")
print(f"scale factor x: {scale_factor_300[0,0]}\nscale factor y: {scale_factor_300[1,1]}\nscale factor z: {scale_factor_300[2,2]}")

print(f"Stats for {Ben_acc_60.get_sensor_name()} sensors:")
print(f"bias x: {bias_60[0][0]}\nbias y: {bias_60[1][0]}\nbias z: {bias_60[2][0]}")
print(f"scale factor x: {scale_factor_60[0,0]}\nscale factor y: {scale_factor_60[1,1]}\nscale factor z: {scale_factor_60[2,2]}")

print(f"Stats for {Ben_acc_5.get_sensor_name()} sensors:")
print(f"bias x: {bias_5[0][0]}\nbias y: {bias_5[1][0]}\nbias z: {bias_5[2][0]}")
print(f"scale factor x: {scale_factor_5[0,0]}\nscale factor y: {scale_factor_5[1,1]}\nscale factor z: {scale_factor_5[2,2]}")


# #### Now assume misalignments

# In[20]:


# Calculate the errors matrix M = [SF + M | b]
M300 = Ben_acc_300.compute_M_errors_matrix()
M60 = Ben_acc_60.compute_M_errors_matrix()
M5 = Ben_acc_5.compute_M_errors_matrix()
print(f"Stats for {Ben_acc_300.get_sensor_name()} sensors:")
print(f"M Matrix:\n{M300}")
print(f"Stats for {Ben_acc_60.get_sensor_name()} sensors:")
print(f"M Matrix:\n{M60}")
print(f"Stats for {Ben_acc_5.get_sensor_name()} sensors:")
print(f"M Matrix:\n{M5}")


# #### Q2.6 - position error in z axis

# In[21]:


position_error = lambda time, bias : (0.5*bias*time**2)

# according to the no misalignment assumption
print("No misalignment case position error:")
print(f"Position error Z after 300sec: {position_error(300, bias_300[2][0])}")
print(f"Position error Z after 60sec: {position_error(60, bias_60[2][0])}")
print(f"Position error Z after 5sec: {position_error(5, bias_5[2][0])}\n")

# according the misalignment case
print("Misalignment case position error:")
print(f"Position error Z after 300sec: {position_error(300, M300[2,3])}")
print(f"Position error Z after 60sec: {position_error(60, M60[2,3])}")
print(f"Position error Z after 5sec: {position_error(5, M5[2,3])}")


# #### Q2.7

# In[22]:


Ben_acc_avg = Accelerometer(name="Ben accelerometer average")
fx_up_avg = (fx_up_300sec + fx_up_60sec + fx_up_5sec) / 3
fx_down_avg = (fx_down_300sec + fx_down_60sec + fx_down_5sec) / 3
fy_up_avg = (fy_up_300sec + fy_up_60sec + fy_up_5sec) / 3
fy_down_avg = (fy_down_300sec + fy_down_60sec + fy_down_5sec) / 3
fz_up_avg = (fz_up_300sec + fz_up_60sec + fz_up_5sec) / 3
fz_down_avg = (fz_down_300sec + fz_down_60sec + fz_down_5sec) / 3

Ben_acc_avg.set_calibration_data(
    fx_up=fx_up_avg, fx_down=fx_down_avg, 
    fy_up=fy_up_avg, fy_down=fy_down_avg, 
    fz_up=fz_up_avg, fz_down=fz_down_avg, 
    g=g_true)

bias_avg = Ben_acc_avg.compute_bias()

print(f"Stats for {Ben_acc_avg.get_sensor_name()} sensors:")
print(f"bias x: {bias_avg[0][0]}\nbias y: {bias_avg[1][0]}\nbias z: {bias_avg[2][0]}")


# ### Gyroscopes

# #### Q3.1

# In[23]:


omega_earth = 7.292115*(10**-5) # rad/s^2
latitude_in_haifa = 32.794046 # degrees
omega_true = omega_earth*np.cos(90-latitude_in_haifa)

# 5sec experiment
wz_up_5sec = get_mean_gyroscopes(ben_gyroscope_rotation_rate_5)
wz_down_5sec = get_mean_gyroscopes(ben_gyroscope_rotation_rate_5_flip)

# 60sec experiment
wz_up_60sec = get_mean_gyroscopes(ben_gyroscope_rotation_rate_60)
wz_down_60sec = get_mean_gyroscopes(ben_gyroscope_rotation_rate_60_flip)

# create sensors objects for the gyros for the different time periods (5, 60 seconds)
Ben_gyr_60 = Gyroscope(name="Ben gyroscope 60 sec")
Ben_gyr_5 = Gyroscope(name="Ben gyroscope 5 sec")

bias_60 = Ben_gyr_60.calc_bias(wz_up_60sec, wz_down_60sec)
scale_factor_60 = Ben_gyr_60.calc_scale_factor(wz_up_60sec, wz_down_60sec, omega_true)
bias_5 = Ben_gyr_5.calc_bias(wz_up_5sec, wz_down_5sec)
scale_factor_5 = Ben_gyr_5.calc_scale_factor(wz_up_5sec, wz_down_5sec, omega_true)

# Print stats for each gyroscope
print(f"Stats for {Ben_gyr_60.get_sensor_name()} sensors:")
print(f"bias z: {bias_60[0][0]}")
print(f"scale factor z: {scale_factor_60[0][0]}\n")

print(f"Stats for {Ben_gyr_5.get_sensor_name()} sensors:")
print(f"bias z: {bias_5[0][0]}")
print(f"scale factor z: {scale_factor_5[0][0]}")



# In[ ]:


ben_gyroscope_rotation_rate_60_flip.head()


# #### Q3.2

# In[24]:


Ben_gyr_32_scenario = Gyroscope(name="Ben gyroscope Q3.2 scenario")

ben_partA_32_scenario = remove_rows_by_time(ben_gyroscope_rotation_rate_32_scenario, 0, 60)
ben_partB_32_scenario = remove_rows_by_time(ben_gyroscope_rotation_rate_32_scenario, 60, 120)

bias = Ben_gyr_32_scenario.calc_bias(get_mean_gyroscopes(ben_partA_32_scenario), get_mean_gyroscopes(ben_partB_32_scenario))
scale_factor = Ben_gyr_32_scenario.calc_scale_factor(get_mean_gyroscopes(ben_partA_32_scenario), get_mean_gyroscopes(ben_partB_32_scenario), omega_true)

print("{}\nBias: \n{}\nScale Factor \n{}".format(Ben_gyr_32_scenario.get_sensor_name(), bias[2][0], scale_factor[2][0]))


# #### Q3.3

# In[25]:


# take average of measurements
wz_up_60sec, wz_down_60sec
wz_up_5sec, wz_down_5sec, omega_true
Ben_gyr_average = Gyroscope(name="Ben gyroscope average")
bias_average = Ben_gyr_average.calc_bias((wz_up_60sec + wz_up_5sec)/2, (wz_down_60sec + wz_down_5sec)/2)
scale_factor_average = Ben_gyr_average.calc_scale_factor((wz_up_60sec + wz_up_5sec)/2, (wz_down_60sec + wz_down_5sec)/2, omega_true)

print(f"Stats for {Ben_gyr_average.get_sensor_name()} sensors:")
print(f"bias z: {bias_average[0][0]}")
print(f"scale factor z: {scale_factor_average[0][0]}")


# ### Accelerometers and Gyroscopes

# #### Q4.2

# In[26]:


# Load experiment data
# Get the data for the accelerometer
f_up_300 = get_mean_accelerations(remove_rows_by_time(ben_accelerometer_and_gyroscope_300_accelerometer, 0, 300 + 0))
f_up_60 = get_mean_accelerations(remove_rows_by_time(ben_accelerometer_and_gyroscope_60_accelerometer, 0, 60 + 0))
f_up_5 = get_mean_accelerations(remove_rows_by_time(ben_accelerometer_and_gyroscope_5_accelerometer, 0, 5 + 0))

# Get the data for the gyroscopes
omega_up_300 = get_mean_gyroscopes(remove_rows_by_time(elad_accelerometer_and_gyroscope_300sec_gyroscope, 0, 300 + 0))
omega_up_60 = get_mean_gyroscopes(remove_rows_by_time(elad_accelerometer_and_gyroscope_60sec_gyroscope, 0, 60 + 0))
omega_up_5 = get_mean_gyroscopes(remove_rows_by_time(elad_accelerometer_and_gyroscope_5sec_gyroscope, 0, 5 + 0))

f_bias_300 = f_up_300 - np.array([[0],[0], [g_true]])
f_bias_60 = f_up_60 - np.array([[0],[0], [g_true]])
f_bias_5 = f_up_5 - np.array([[0],[0], [g_true]])

omega_bias_300 = omega_up_300 - np.array([[0],[0], [omega_true]])
omega_bias_60 = omega_up_60 - np.array([[0],[0], [omega_true]])
omega_bias_5 = omega_up_5 - np.array([[0],[0], [omega_true]])

print(f"Stats for 300sec:")
print(f"accel bias x: {f_bias_300[0][0]}\naccel bias y: {f_bias_300[1][0]}\naccel bias z: {f_bias_300[2][0]}")
print(f"gyro bias x: {omega_bias_300[0][0]}\ngyro bias y: {omega_bias_300[1][0]}\ngyro bias z: {omega_bias_300[2][0]}")
print(f"Stats for 60sec:")
print(f"accel bias x: {f_bias_60[0][0]}\naccel bias y: {f_bias_60[1][0]}\naccel bias z: {f_bias_60[2][0]}")
print(f"gyro bias x: {omega_bias_60[0][0]}\ngyro bias y: {omega_bias_60[1][0]}\ngyro bias z: {omega_bias_60[2][0]}")
print(f"Stats for 5sec:")
print(f"accel bias x: {f_bias_5[0][0]}\naccel bias y: {f_bias_5[1][0]}\naccel bias z: {f_bias_5[2][0]}")
print(f"gyro bias x: {omega_bias_5[0][0]}\ngyro bias y: {omega_bias_5[1][0]}\ngyro bias z: {omega_bias_5[2][0]}")


# #### Q4.3

# In[27]:


Ben_shake_accel = remove_rows_by_time(ben_accelerometer_and_gyroscope_shake_accelerometer, 0, 30 + 0)
Ben_shake_gyro = remove_rows_by_time(ben_accelerometer_and_gyroscope_shake_gyroscope, 0, 30 + 0)

Ben_shake = Sensor(name="Ben shake accelerometer")
Ben_shake.plot_time_series(Ben_shake_accel["Time (s)"].to_numpy(), Ben_shake_accel[["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)"]].to_numpy(), "Accelerations (m/s^2)", "Time (s)")


# #### Q4.4

# In[28]:


# Load experiment data
# Get the data for the accelerometer
f_up_300 = get_mean_accelerations(remove_rows_by_time(ben_accelerometer_and_gyroscope_aftershake_300_accelerometer, START_TIME_BUFFER, 300 + START_TIME_BUFFER))
f_up_60 = get_mean_accelerations(remove_rows_by_time(ben_accelerometer_and_gyroscope_aftershake_60_accelerometer, START_TIME_BUFFER, 60 + START_TIME_BUFFER))
f_up_5 = get_mean_accelerations(remove_rows_by_time(ben_accelerometer_and_gyroscope_aftershake_5_accelerometer, START_TIME_BUFFER, 5 + START_TIME_BUFFER))

# Get the data for the gyroscopes
omega_up_300 = get_mean_gyroscopes(remove_rows_by_time(ben_accelerometer_and_gyroscope_aftershake_300_gyroscope, START_TIME_BUFFER, 300 + START_TIME_BUFFER))
omega_up_60 = get_mean_gyroscopes(remove_rows_by_time(ben_accelerometer_and_gyroscope_aftershake_60_gyroscope, START_TIME_BUFFER, 60 + START_TIME_BUFFER))
omega_up_5 = get_mean_gyroscopes(remove_rows_by_time(ben_accelerometer_and_gyroscope_aftershake_5_gyroscope, START_TIME_BUFFER, 5 + START_TIME_BUFFER))

f_bias_300 = f_up_300 - np.array([[0],[0], [g_true]])
f_bias_60 = f_up_60 - np.array([[0],[0], [g_true]])
f_bias_5 = f_up_5 - np.array([[0],[0], [g_true]])

omega_bias_300 = omega_up_300 - np.array([[0],[0], [omega_true]])
omega_bias_60 = omega_up_60 - np.array([[0],[0], [omega_true]])
omega_bias_5 = omega_up_5 - np.array([[0],[0], [omega_true]])

print(f"Stats for 300sec:")
print(f"accel bias x: {f_bias_300[0][0]}\naccel bias y: {f_bias_300[1][0]}\naccel bias z: {f_bias_300[2][0]}")
print(f"gyro bias x: {omega_bias_300[0][0]}\ngyro bias y: {omega_bias_300[1][0]}\ngyro bias z: {omega_bias_300[2][0]}")
print(f"Stats for 60sec:")
print(f"accel bias x: {f_bias_60[0][0]}\naccel bias y: {f_bias_60[1][0]}\naccel bias z: {f_bias_60[2][0]}")
print(f"gyro bias x: {omega_bias_60[0][0]}\ngyro bias y: {omega_bias_60[1][0]}\ngyro bias z: {omega_bias_60[2][0]}")
print(f"Stats for 5sec:")
print(f"accel bias x: {f_bias_5[0][0]}\naccel bias y: {f_bias_5[1][0]}\naccel bias z: {f_bias_5[2][0]}")
print(f"gyro bias x: {omega_bias_5[0][0]}\ngyro bias y: {omega_bias_5[1][0]}\ngyro bias z: {omega_bias_5[2][0]}")

