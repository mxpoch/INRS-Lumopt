######## IMPORTS ########
# General purpose imports
import numpy as np
import os
import sys
import scipy as sp
import time
#import sys
#sys.path.append( "C:\\Program Files\\Lumerical\\v202\\api\\python")
import lumapi
import pdb 
import math

# Optimization specific imports
from lumopt import CONFIG

from lumopt.geometries.topology import TopologyOptimization2D
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.utilities.wavelengths import Wavelengths

######## RUNS TOPOLOGY OPTIMIZATION OF A 2D STRUCTURE ########
def runSim(params, eps_min, eps_max, x_pos, y_pos, filter_R):

    ######## DEFINE A 2D TOPOLOGY OPTIMIZATION REGION ########
    geometry = TopologyOptimization2D(params=params, eps_min=eps_min, eps_max=eps_max, x=x_pos, y=y_pos, min_feature_size=28*1e-6, filter_R=filter_R)

    ######## DEFINE FIGURE OF MERIT ########
    # The base simulation script defines a field monitor named 'fom' at the point where we want to modematch to the fundamental TE mode
    fom_N = ModeMatch(monitor_name = 'fom_N', mode_number = 1, direction = 'Forward', target_T_fwd=lambda wl: np.ones(wl.size), norm_p = 2)
    fom_S = ModeMatch(monitor_name = 'fom_S', mode_number = 2, direction = 'Forward', target_T_fwd=lambda wl: np.ones(wl.size), norm_p = 2)

    ######## DEFINE OPTIMIZATION ALGORITHM ########
    #optimizer = ScipyOptimizers(max_iter=50, method='L-BFGS-B', scaling_factor=1, pgtol=1e-6, ftol=1e-4, target_fom=0.5, scale_initial_gradient_to=0.25)
    optimizer = ScipyOptimizers(max_iter=34, method='L-BFGS-B', scaling_factor=1, pgtol=1e-5, ftol=1e-5, scale_initial_gradient_to=0.0)
    
    wavelengths = Wavelengths(start = 289.792*1e-6, stop = 309.793*1e-6, points = 5)
    opt_N = Optimization(base_script=base_geom, wavelengths = wavelengths, fom=fom_N, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=True, plot_history=False, store_all_simulations=False)
    opt_S = Optimization(base_script=base_geom, wavelengths = wavelengths, fom=fom_S, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=True, plot_history=False, store_all_simulations=False)
    
    opt = opt_N+opt_S
    ######## RUN THE OPTIMIZER ########
    opt.run()
    
    # EXTRACTING THE CONTOURS
    
# GEOMETRY
def base_geom(fdtd):
    px_size = 2 #micron
    x = 520
    y = 520
    # SIM PARAMS
    x_points=int(x/px_size)+1
    x_pos = np.linspace(-x/2,x/2,x_points)*1e-6
    
    opt_size_x = max(x_pos)-min(x_pos)
    opt_size_y = y*1e-6
     
    size_x = opt_size_x+130e-6
    size_y = opt_size_y+130e-6

    out_wg_dist = 120e-6
    wg_width = 80e-6
    mode_width = 3 * wg_width

    wg_index = round(math.sqrt(11.9),2)
    bg_index = 1

    dx = 0.5e-5

    # INPUT WAVEGUIDE
    fdtd.addrect()
    fdtd.set('name', 'input wg')
    fdtd.set('x min', -size_x)
    fdtd.set('x max', -opt_size_x / 2)
    fdtd.set('y', 0)
    fdtd.set('y span', wg_width)
    fdtd.set('z', 0)
    fdtd.set('z span', 220e-7)
    fdtd.set('material', '<Object defined dielectric>')
    fdtd.set('index', wg_index)

    # OUTPUT WAVEGUIDES
    fdtd.addrect()
    fdtd.set('name', 'output wg N')
    fdtd.set('y min', opt_size_y / 2 )
    fdtd.set('y max', size_y)
    fdtd.set('x', 0)
    fdtd.set('x span', wg_width)
    fdtd.set('z', 0)
    fdtd.set('z span', 220e-7)
    fdtd.set('material', '<Object defined dielectric>')
    fdtd.set('index', wg_index)

    fdtd.addrect()
    fdtd.set('name', 'output wg S')
    fdtd.set('y max', -opt_size_y / 2 )
    fdtd.set('y min', -size_y)
    fdtd.set('x', 0)
    fdtd.set('x span', wg_width)
    fdtd.set('z', 0)
    fdtd.set('z span', 220e-7)
    fdtd.set('material', '<Object defined dielectric>')
    fdtd.set('index', wg_index)
 

    # SOURCE
    fdtd.addmode()
    fdtd.set('direction', 'Forward')
    fdtd.set('injection axis', 'x-axis')
    fdtd.set('x', -opt_size_x / 2 - 40e-6)
    fdtd.set('y', 0)
    fdtd.set('y span', size_y)
    fdtd.set('z', 0)
    fdtd.set('z span', 1e-6)
    fdtd.set('center wavelength', 0.0002997925)
    fdtd.set('wavelength span', 20e-6)
    fdtd.set('mode selection', 'fundamental TE mode')

    # FDTD
    fdtd.addfdtd()
    fdtd.set('dimension', '2D')
    fdtd.set('background index', bg_index)
    fdtd.set('mesh accuracy',6)
    fdtd.set('x min', -size_x / 2)
    fdtd.set('x max', size_x / 2)
    fdtd.set('y min', -size_y / 2)
    fdtd.set('y max', size_y / 2)
    fdtd.set('auto shutoff min', 1e-6)
    fdtd.set('simulation time', 1e-10)

    # OPTIMIZATION FIELDS MONITOR IN OPTIMIZABLE REGION
    fdtd.addpower()
    fdtd.set('name', 'opt_fields')
    fdtd.set('monitor type', '2D Z-normal')
    fdtd.set('x min', -opt_size_x/2)
    fdtd.set('x max', opt_size_x/2)
    fdtd.set('y min', -opt_size_y/2)
    fdtd.set('y max', opt_size_y/2)

    # FOM FIELDS
    fdtd.addpower()
    fdtd.set('name', 'fom_N')
    fdtd.set('monitor type', '2D Y-normal')
    fdtd.set('x', 0)
    fdtd.set('y', opt_size_y/2 + 20e-6)
    fdtd.set('x span', opt_size_y/2)
    fdtd.setglobalmonitor("frequency points", 100)

    fdtd.addmesh()
    fdtd.set('name', 'fom_mesh_N')
    fdtd.set('override x mesh', True)
    fdtd.set('dx', dx)
    fdtd.set('override y mesh', False)
    fdtd.set('override z mesh', False)
    fdtd.set('x', 0)
    fdtd.set('x span', 5e-6)
    fdtd.set('y', opt_size_y/2 + 20e-6)
    fdtd.set('x span', opt_size_x/2)
    
    fdtd.addpower()
    fdtd.set('name', 'fom_S')
    fdtd.set('monitor type', '2D Y-normal')
    fdtd.set('x', 0)
    fdtd.set('y', -opt_size_y/2 - 20e-6)
    fdtd.set('x span', opt_size_y/2)
    fdtd.setglobalmonitor("frequency points", 100)

    fdtd.addmesh()
    fdtd.set('name', 'fom_mesh_S')
    fdtd.set('override x mesh', True)
    fdtd.set('dx', dx)
    fdtd.set('override y mesh', False)
    fdtd.set('override z mesh', False)
    fdtd.set('x', 0)
    fdtd.set('x span', 5e-6)
    fdtd.set('y', -opt_size_y/2 - 20e-6)
    fdtd.set('x span', opt_size_x/2)

    # For visualization later
    fdtd.addindex()
    fdtd.set('name', 'global_index')
    fdtd.set('x min', -size_x / 2)
    fdtd.set('x max', size_x / 2)
    fdtd.set('y min', -size_y / 2)
    fdtd.set('y max', size_y / 2)

if __name__ == '__main__':
    size_x = 520		#< Length of the device (in nm). Longer devices typically lead to better performance
    delta_x = 2		    #< Size of a pixel along x-axis (in nm)

    size_y = 520		#< Since we use symmetry, this is only have the extent along the y-axis (in nm
    delta_y = 2	        #< Size of a pixel along y-axis (in nm)
	
    filter_R = 28	    #< Radius of the smoothing filter which removes small features and sharp corners (in nm)
   
    eps_max = 11.9	    #< Effective permittivity for a Silicon waveguide with a thickness of 220nm
    eps_min = 1	        #< Permittivity of the SiO2 cladding

    x_points=int(size_x/delta_x)+1
    y_points=int(size_y/delta_y)+1

    x_pos = np.linspace(-size_x/2,size_x/2,x_points)*1e-6
    y_pos = np.linspace(-size_y/2,size_y/2,y_points)*1e-6

    ## Alternative initial conditions
    #initial_cond = np.ones((x_points,y_points))       #< Start with the domain filled with eps_max
    initial_cond = 0.5*np.ones((x_points,y_points))   #< Start with the domain filled with (eps_max+eps_min)/2
    #initial_cond = np.zeros((x_points,y_points))      #< Start with the domain filled with eps_min         
    runSim(initial_cond, eps_min, eps_max, x_pos, y_pos, filter_R*1e-6)
    
    #with lumapi.FDTD(hide=False) as fdtd:
        ## help(fdtd.addvarfdtd)
        #base_geom(fdtd)
        ## just enough time to take a look around
        #time.sleep(30)