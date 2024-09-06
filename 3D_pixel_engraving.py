import numpy as np
import os
import sys
import scipy as sp
import time
import lumapi
from lumopt.geometries.topology import TopologyOptimization2D, TopologyOptimization3DLayered
import matplotlib.pyplot as plt

# GEOMETRY
def base_geom(fdtd):
    plate_thickness = 0.001
    gap = 0.0006
    # in mm
    # SIM PARAMS 
    opt_size_x = 20e-3
    opt_size_y = 20e-3
    opt_size_z = 0.00028
     
    size_x = opt_size_x+10e-3
    size_y = opt_size_y+10e-3
    size_z = opt_size_z + plate_thickness*2 + gap

    bg_index = 1

    dx = 1100e-7

    # SOURCE
    fdtd.addmode()
    fdtd.set('direction', 'Forward')
    fdtd.set('injection axis', 'x-axis')
    fdtd.set('x', -opt_size_x / 2 + 1e-4)
    fdtd.set('y', 0)
    fdtd.set('y span', opt_size_y)
    #fdtd.set('y span', wg_width)
    fdtd.set('z', 0)
    fdtd.set('z span', size_z)
    fdtd.set('center wavelength', 0.0011103424)
    fdtd.set('wavelength span', 0.0005)
    fdtd.set('mode selection', 'user select')
    fdtd.set('selected mode number', 3)

    # FDTD
    fdtd.addfdtd()
    fdtd.set('dimension', '3D')
    fdtd.set('background index', bg_index)
    
    fdtd.set('mesh type', 'custom non-uniform')
    fdtd.set('define x mesh by', 'maximum mesh step')
    fdtd.set('define y mesh by', 'maximum mesh step')
    fdtd.set('define z mesh by', 'maximum mesh step')
    fdtd.set('dx', 200*1e-6)
    fdtd.set('dy', 200*1e-6)
    fdtd.set('dz', 50*1e-6)    
    fdtd.set('x span', size_x)
    fdtd.set('x', 0)
    fdtd.set('y span', size_y)
    fdtd.set('y', 0)
    fdtd.set('z span', size_z)
    fdtd.set('z', 0)
    fdtd.set('auto shutoff min', 1e-5)
    fdtd.set('simulation time', 1e-8)
    
    # ADDING THE PARALLEL PLATES
    
    # ENGRAVING THE PLATE
    # calculating the engraving distances 
    engraving_dist = param_to_engraving_dist(70, 300, [100,100])
    engraving_dist = engraving_dist.transpose()
    px_width = opt_size_x / engraving_dist.shape[0]
    
    # for plotting the engraving distances
    plt.imshow(engraving_dist, interpolation='none')
    plt.show()
    
    # placing the rectangles    
    for iy, ix in np.ndindex(engraving_dist.shape):
        fdtd.addrect()
        # mapping index to position
        # starting from the left + half the width of the pixel + number of pixel widths
        fdtd.set('x',-10e-3 + px_width/2 + ix*px_width)
        fdtd.set('y',-10e-3 + px_width/2 + iy*px_width)
        fdtd.set('z',-engraving_dist[ix,iy]/2 - 0.0006/2)
        fdtd.set('x span', px_width)
        fdtd.set('y span', px_width)
        fdtd.set('z span', engraving_dist[ix,iy])
        fdtd.set('name', f'{ix},{iy}')
        fdtd.set('override mesh order from material database', 1)
        fdtd.set('mesh order', 1)
        fdtd.set('material', '<Object defined dielectric>')
        fdtd.set('index', 1)
    
    # measurements are in meters unless variable    
    fdtd.addrect()
    fdtd.set('name', 'bottom plate')
    fdtd.set('x', 0)
    fdtd.set('y',0)
    fdtd.set('z', -plate_thickness/2-gap/2)
    fdtd.set('z span', 0.001)
    fdtd.set('y span', opt_size_y)
    fdtd.set('x span', opt_size_x)
    fdtd.set('material', 'PEC (Perfect Electrical Conductor)')
   
    fdtd.addrect()
    fdtd.set('x',0)
    fdtd.set('y',0)
    fdtd.set('name', 'top plate')
    fdtd.set('z', plate_thickness/2+gap/2)
    fdtd.set('z span', 0.001)
    fdtd.set('y span', opt_size_y)
    fdtd.set('x span', opt_size_x)
    fdtd.set('material', 'PEC (Perfect Electrical Conductor)')
    
    # OPTIMIZATION FIELDS MONITOR IN OPTIMIZABLE REGION
    fdtd.addpower()
    fdtd.set('name', 'opt_fields')
    fdtd.set('monitor type', '2D Z-normal')
    fdtd.set('spatial interpolation', 'nearest mesh cell')
    fdtd.set('x span', opt_size_x)
    fdtd.set('x', 0)
    fdtd.set('y span', opt_size_y)
    fdtd.set('y', 0)

    # FOM FIELDS
    fdtd.addpower()
    fdtd.set('name', 'fom')
    fdtd.set('monitor type', '2D X-normal')
    fdtd.set('y', 0)
    fdtd.set('x', opt_size_x/2 + 1e-3)
    fdtd.set('y span', size_y)
    fdtd.set('z', 0)
    fdtd.set('z span', size_z)
    fdtd.setglobalmonitor("frequency points", 100) 
    
    fdtd.addpower()
    fdtd.set('name', 'mode_monitor')
    fdtd.set('monitor type', '2D Y-normal')
    fdtd.set('y', 0)
    fdtd.set('x', 0)
    fdtd.set('x span', opt_size_x)
    fdtd.set('z', 0)
    fdtd.set('z span', size_z)
    fdtd.setglobalmonitor("frequency points", 100) 

def param_to_engraving_dist(opt_num:int, iter_num:int, matrix_param):
    # pulling the params from file
    path2d = "D:\\lumerical\\python_only\\opts_{}".format(opt_num)
    geom2d = TopologyOptimization2D.from_file(os.path.join(path2d, 'parameters_{}.npz'.format(iter_num) ), filter_R=1e-4, eta=0.5)
    starting_params = geom2d.last_params
    
    # mapping the param space to length space
    # first converting from param to eps
    eps_max = 0.621	
    eps_min = 0.202
    
    param_to_eps = lambda x: (eps_max-eps_min)*x+eps_min    
    v_pte = np.vectorize(param_to_eps)    
    
    eps_param = v_pte(starting_params)
    
    # converting from eps to distance
    f = 0.27*10**12 # 0.27THz -> Hz
    sol = 299792458 # speed of light m/s
    dist_param = sol/(2*f*(np.sqrt(1-eps_param)))
    
    # converting from plate separation to engraving distance
    eng_dist_param = np.abs(np.amin(dist_param) - dist_param) 
    
    # converting the resolution of the matrix 
    def shrink(data, rows, cols):
        # reslicing data
        data = data[:data.shape[0]-1, :data.shape[1]-1]
        return data.reshape(rows, int(data.shape[0]/rows), cols, int(data.shape[1]/cols)).mean(axis=1).mean(axis=2) 
    
    print(eng_dist_param, matrix_param[1], matrix_param[0])
    derezzed_eng_dist = shrink(eng_dist_param, matrix_param[1], matrix_param[0]) 
    return derezzed_eng_dist

if __name__ == "__main__":
    # populating the simulation
    with lumapi.FDTD(hide=False) as fdtd:
        # help(fdtd.addvarfdtd)
        base_geom(fdtd)
        # just enough time to take a look around
        time.sleep(172800)
    