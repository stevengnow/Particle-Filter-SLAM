# %%
import numpy as np
from world_coord import world_coordinates
import datetime
from pr2_utils import bresenham2D, mapCorrelation
from find_nearest import find_nearest
import matplotlib.pyplot as plt
import random
def particle_filter(N,m,alpha, mu, v_noise, omega_noise, enc_data, fog_data, lidar_data):
    '''

    '''
    #Initializations
    xmin = -1400
    xmax = 1400
    ymin = -1400
    ymax = 1400
    res = 1
    MAP_sizex  = int(np.ceil((xmax - xmin) / res + 1))        #cells
    MAP_sizey  = int(np.ceil((ymax - ymin) / res + 1))
    x_range = np.arange(-0.4,0.4+0.1,0.1)
    y_range = np.arange(-0.4,0.4+0.1,0.1)
    x_im = np.arange(xmin,xmax+res,res) 
    y_im = np.arange(ymin,ymax+res,res)
    begin_time = datetime.datetime.now()
    MAP = np.zeros((MAP_sizex,MAP_sizey), dtype = np.int8)
    traj = np.empty([encoder_data.shape[0]+1,3])

    enc_cnt = 1
    alpha_idx = np.arange(0,N)
    #Calculate initial Velocity
    tao = (enc_data[enc_cnt][0] - enc_data[enc_cnt-1][0])/(10**9)
    v_left = (np.pi*0.623479*(enc_data[enc_cnt][1]- enc_data[enc_cnt-1][1]))/(4096*tao)
    v_right = (np.pi*0.622806*(enc_data[enc_cnt][2]- enc_data[enc_cnt-1][2]))/(4096*tao)
    v_t = (v_left + v_right) / 2
    noise_v = v_t + v_noise   # add noise
    for i in range(1,lidar_data.shape[0]*10):
        # if i % 100000 == 0: #for saving
        #     print(i)
        #     print('Time so far | ', datetime.datetime.now() - begin_time)
        #     np.save('traj_25/traj_'+ str(i), traj)
        #     # np.save('MAP/building_map_'+ str(i), MAP)
        # PREDICT
        if enc_data[enc_cnt][0] > fog_data[i][0]:
            tao_fog = (fog_data[i][0] - fog_data[i-1][0])/(10**9)
            mu = mu + tao_fog*(np.stack((noise_v*np.cos(mu[2,:]),noise_v*np.sin(mu[2,:]),(fog_data[i][3])/tao_fog + omega_noise)))         
        else:
        # GET LIDAR PREPARE FOR UPDATE AND MAP
            lidar_idx = np.where(lidar_data[:,0] == find_nearest(lidar_data[:,0],encoder_data[enc_cnt][0]))
            ranges = lidar_data[lidar_idx[0][0],:]   
            angles = np.linspace(-5, 185, 287) / 180 * np.pi                                
            indValid = np.logical_and((ranges < 75),(ranges> 2))
            ranges = ranges[indValid]
            angles = angles[indValid]
            e_matrix = np.zeros((mu.shape[1]))  # used for determining greatest alpha

        # MAP CORRELATION
            for particle in range(0,mu.shape[1]):
                wc = world_coordinates(ranges,angles,mu[:,particle])
                e_matrix[particle] = np.exp(np.amax(mapCorrelation(MAP,x_im,y_im, wc.T, x_range,y_range)))     
            
            alpha = e_matrix * alpha / np.sum(e_matrix * alpha)
            max_idx = np.argmax(alpha)

            wc_max = world_coordinates(ranges,angles,mu[:,max_idx])  
            traj[enc_cnt] = mu[:,max_idx]
        # MAPPING
            for j in range(0,wc_max.shape[0]):
                xis = np.ceil((wc_max[j][0] - xmin) / res ).astype(np.int16)-1
                yis = np.ceil((wc_max[j][1] - ymin) / res ).astype(np.int16)-1
                mux = np.ceil((mu[0,max_idx] - xmin) / res ).astype(np.int16)-1
                muy = np.ceil((mu[1,max_idx] - ymin) / res ).astype(np.int16)-1
                x = bresenham2D(mux,muy,xis, yis)[0]
                y = bresenham2D(mux,muy,xis, yis)[1]
               
                m[xis,yis] = m[xis,yis] + np.log(4)
                if m[xis,yis] > 0:
                    MAP[xis,yis] = 1

                m[x[:-1].astype(int), y[:-1].astype(int)] = m[x[:-1].astype(int), y[:-1].astype(int)] - np.log(4)

                for k in range(0,x.shape[0]-1):
                    if m[int(x[k]),int(y[k])] < 0:
                        MAP[int(x[k]),int(y[k])] = -1

           
        # RESAMPLING
            if 1 / (np.sum(alpha**2)) <= N*0.2:
                choices = random.choices(alpha_idx, weights = alpha, k = N)
                mu = mu[:, choices]
                alpha = np.ones(N)*1/N

        #Recalculate the velocity
            enc_cnt += 1
            tao = (enc_data[enc_cnt][0] - enc_data[enc_cnt-1][0])/(10**9)
            v_left = (np.pi*0.623479*(enc_data[enc_cnt][1]- enc_data[enc_cnt-1][1]))/(4096*tao)
            v_right = (np.pi*0.622806*(enc_data[enc_cnt][2]- enc_data[enc_cnt-1][2]))/(4096*tao)
            v_t = (v_left + v_right) / 2
            noise_v = v_t + v_noise  # add noise
            tao_fog = (fog_data[i][0] - fog_data[i-1][0])/(10**9)
            mu = mu + tao_fog*(np.stack((noise_v*np.cos(mu[2,:]),noise_v*np.sin(mu[2,:]),fog_data[i][3]/tao_fog + omega_noise)))     

    print('Time to complete particle filter | ', datetime.datetime.now() - begin_time)
    return m, MAP, traj


lidar_data = np.load('lidar_data.npy')
fog_data = np.load('fog_data.npy')
encoder_data = np.load('encoder_data.npy')
N = 25
alpha = np.ones(N)*1/N                                  #initialize particle weights
v_noise = np.random.normal(0,0.01,N)
omega_noise = np.random.normal(0,0.001,N)                #number of particles
mu = np.zeros((3,N))                                    #initialize particle set
m = np.load('initialmap.npy')
log_map, MAP, traj = particle_filter(N,m,alpha, mu,v_noise, omega_noise, encoder_data, fog_data, lidar_data)

# # %% to plot traj
# traj = np.load('traj_25/traj_end.npy')
# traj = traj[1:]
# x_coord, y_coord = traj[:,:2].T
# x_coord = x_coord[x_coord != 0]
# y_coord = y_coord[y_coord != 0]
# plt.plot(x_coord, y_coord)
# # %% to plot map
# build_map = np.load('1_dead_rec/building_map_1100000.npy')
# build_map = np.negative(build_map)
# build_map = np.flip(build_map.T, axis = 0)
# plt.imshow(build_map, cmap = 'gray', vmin = -1, vmax = 1)
# plt.title("Occupancy grid map")
# plt.show()
# input('Press ENTER to exit')

