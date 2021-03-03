# %% Imports
import numpy as np
import datetime
from pr2_utils import bresenham2D
import matplotlib.pyplot as plt
# from Particle_filter import particle_filter, world_coordinates
from world_coord import world_coordinates
#%% Load in the Data
lidar_data = np.load('lidar_data.npy')
fog_data = np.load('fog_data.npy')
encoder_data = np.load('encoder_data.npy')
# %% Initializations
angles = np.linspace(-5, 185, 286) / 180 * np.pi
ranges = lidar_data[0, 1:]                               #first scan
indValid = np.logical_and((ranges < 75),(ranges> 2))
xmin = -1400
xmax = 1400
ymin = -1400
ymax = 1400

res = 1
ranges = ranges[indValid]                                 #find valid ranges
angles = angles[indValid]
MAP_sizex  = int(np.ceil((xmax - xmin) / res + 1))        #cells
MAP_sizey  = int(np.ceil((ymax - ymin) / res + 1))
m = np.zeros((MAP_sizex,MAP_sizey), dtype = 'float64')                       #initialize map 1000x1000
N = 25                                                   #number of particles
mu = np.zeros((3,N))                                      #initialize particle set
alpha = np.ones(N)*1/N                                    #initialize particle weights
coord = np.zeros((3,ranges.shape[0]))                     #coordinates (top row x values, bottom row y values)
l_to_b = np.array(([0.00130201, 0.796097, 0.605167],[0.999999 ,-0.000419027, -0.00160026],[-0.00102038, 0.605169 ,-0.796097 ]))  #rotation matrix: lidar to body
lidar_t = np.array((0.8349, -0.0126869, 1.76416))
position = np.array((0,0,0))                        #start position
# %% Find the World Coordinates of the first Scan
wc = world_coordinates(ranges, angles, position)

# %% Update Initial Log Odds
begin_time = datetime.datetime.now()

for i in range(0,wc.shape[0]):
    xis = np.ceil((wc[i][0] - xmin) / res ).astype(np.int16)-1
    yis = np.ceil((wc[i][1] - ymin) / res ).astype(np.int16)-1
    bres = bresenham2D(1400,1400,xis, yis)
    x = bres[0]
    y = bres[1]
    m[xis][yis] = m[xis][yis] + np.log(4)
    m[x[:-1].astype(int), y[:-1].astype(int)] = m[x[:-1].astype(int), y[:-1].astype(int)] - np.log(4)
np.save('initialmap', m)  
print('Time to complete initial map | ', datetime.datetime.now() - begin_time)

# %% Plot the Initial Map
plt.imshow(m, cmap="RdBu")
plt.title("Occupancy grid map")
plt.show()
input('Press ENTER to exit')

# %% Dead-Reckoning
enc_cnt = 1
traj = np.empty([encoder_data.shape[0]*10,3])
begin_time = datetime.datetime.now()
tao = (encoder_data[enc_cnt][0] - encoder_data[enc_cnt-1][0])/(10**9)
v_left = (np.pi*0.623479*(encoder_data[enc_cnt][1]- encoder_data[enc_cnt-1][1]))/4096/tao
v_right = (np.pi*0.622806*(encoder_data[enc_cnt][2]- encoder_data[enc_cnt-1][2]))/4096/tao
v_t = (v_left + v_right) /2 
for i in range(1,encoder_data.shape[0]*10):       
    if i % 100000 == 0:
            print(i)
            print('Time so far | ', datetime.datetime.now() - begin_time)
            np.save('traj_dead/traj_'+ str(i), traj)
    if encoder_data[enc_cnt][0] > fog_data[i][0]:
        tao_fog = (fog_data[i][0] - fog_data[i-1][0])/(10**9)
        position = position + tao_fog*np.array((v_t*np.cos(position[2]),v_t*np.sin(position[2]),fog_data[i][-1]/tao_fog))
        traj[i] = position
    else:  
        enc_cnt += 1
        tao = (encoder_data[enc_cnt][0] - encoder_data[enc_cnt-1][0])/(10**9)
        v_left = (np.pi*0.623479*(encoder_data[enc_cnt][1]- encoder_data[enc_cnt-1][1]))/4096/tao
        v_right = (np.pi*0.622806*(encoder_data[enc_cnt][2]- encoder_data[enc_cnt-1][2]))/4096/tao
        v_t = (v_left + v_right) /2  
        tao_fog = (fog_data[i][0] - fog_data[i-1][0])/(10**9)
        position = position + tao_fog*np.array((v_t*np.cos(position[2]),v_t*np.sin(position[2]),fog_data[i][-1]/tao_fog))
        traj[i] = position
traj = traj[1:]
print('Time to complete dead-reckoning | ', datetime.datetime.now() - begin_time)
#%% Plot Dead-Reckoning
x_coord, y_coord = traj[:,:2].T
plt.plot(x_coord,y_coord)
plt.title('Dead Reckoning Estimated Vehicle Trajectory')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.text(0,0,'Start')
plt.text(traj[-1][0],traj[-1][1],'End')
plt.show()
input('Press ENTER to exit')



