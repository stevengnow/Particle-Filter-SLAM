import numpy as np
def world_coordinates(ranges, angles, mu):
    '''
    find world coordinates of an input LiDAR scan
    '''
    lidar_t = np.array((0.8349, -0.0126869, 1.76416))
    l_to_b = np.array(([0.00130201, 0.796097, 0.605167],[0.999999 ,-0.000419027, -0.00160026],[-0.00102038, 0.605169 ,-0.796097 ]))  #rotation matrix: lidar to body
    b_to_w = np.array(([np.cos(mu[2]), -np.sin(mu[2]), 0],[np.sin(mu[2]), np.cos(mu[2]), 0],[0,0,1]))
    xs = ranges*np.cos(angles)
    ys = ranges*np.sin(angles)
    coord = np.vstack((xs,ys,np.ones(ranges.shape[0])))  # should these be 0?

    body_coord = np.dot(coord.T,l_to_b) + lidar_t
    world_coord = np.dot( body_coord, b_to_w) + mu  # + pose

    return world_coord[:,:-1]


