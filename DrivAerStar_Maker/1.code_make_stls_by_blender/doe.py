import matplotlib.pyplot as plt
import numpy as np


dim = 20  




num = 8000  
sample = np.ones((num, dim))
sample[:, 0 ] = np.random.uniform(0.8, 1.2, num)  
sample[:, 1 ] = np.random.uniform(-0.1, 0.1, num)  
sample[:, 2 ] = np.random.uniform(-0.1, 0.1, num)  
sample[:, 3 ] = np.random.uniform(-8, 8, num)  
sample[:, 4 ] = np.random.uniform(-0.1, 0.1, num)  
sample[:, 5 ] = np.random.uniform(-0.05, 0.05, num)  
sample[:, 6 ] = np.random.uniform(-0.05, 0.05, num)  
sample[:, 7 ] = np.random.uniform(-0.05, 0.05, num)  
sample[:, 8 ] = np.random.uniform(-0.05, 0.05, num)  
sample[:, 9 ] = np.random.uniform(-0.05, 0.05, num)  
sample[:, 10] = np.random.uniform(-0.05, 0.05, num)  
sample[:, 11] = np.random.uniform(-8, 8, num)  
sample[:, 12] = np.random.uniform(-0.05, 0.05, num)  
sample[:, 13] = np.random.uniform(-0.05, 0.05, num)  
sample[:, 14] = np.random.uniform(-8, 8, num)  
sample[:, 15] = np.random.uniform(-8, 8, num)  
sample[:, 16] = np.random.uniform(-8, 8, num)  
sample[:, 17] = np.random.uniform(-8, 8, num)  
sample[:, 18] = np.random.uniform(-0.013, 0.013, num)  
sample[:, 19] = np.random.uniform(-0.015, 0.015, num)  


headers = [
    "car_size", "car_width", "car_len", "ramp_angle",
    "front_bumper_length", "wind_screen_x", "wind_screen_z",
    "side_mirrors_x", "side_mirrors_z", "rear_window_x",
    "rear_window_z", "trunklid_angle", "trunklid_x",
    "trunklid_z", "diffusor_angle","car_green_house_angle",
    "car_front_hood_angle","car_air_intake_angle","tires_diameter","tires_width"
]

np.savetxt('lhs_parameters_Notch_v3.csv', sample, delimiter=',', header=','.join(headers), comments='')