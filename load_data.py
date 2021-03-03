import numpy as np
from numpy import genfromtxt
lidar_data = genfromtxt('data/sensor_data/lidar.csv', delimiter = ',')
np.save('lidar_data', lidar_data)
fog_data = genfromtxt('data/sensor_data/fog.csv', delimiter = ',')
np.save('fog_data', fog_data)
encoder_data = genfromtxt('data/sensor_data/encoder.csv', delimiter = ',')
np.save('encoder_data', encoder_data)
