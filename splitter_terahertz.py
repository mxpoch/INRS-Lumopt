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

#from lumopt.geometries.topology import TopologyOptimization2D
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.utilities.wavelengths import Wavelengths



#### SPECIAL IMPORT FOR TOPO2D -- skip to bottom of file for simulation code
from lumopt.geometries.geometry import Geometry
from lumopt.utilities.materials import Material
from lumopt.lumerical_methods.lumerical_scripts import set_spatial_interp, get_eps_from_sim

import lumapi
import numpy as np
import scipy as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

eps0 = sp.constants.epsilon_0

class TopologyOptimization2DParameters(Geometry):

    def __init__(self, params, eps_min, eps_max, x, y, z, filter_R, eta, beta, eps=None, min_feature_size = 0):
        self.last_params=params
        self.eps_min=eps_min
        self.eps_max=eps_max
        self.eps = eps
        self.x=x
        self.y=y
        self.z=z
        self.bounds=[(0.,1.)]*(len(x)*len(y))
        self.filter_R = filter_R
        self.eta = eta

        if (min_feature_size<0) or (min_feature_size>(2*filter_R)):
            raise UserWarning('Value of min_feature_size must be between 0 and 2*filter_R ({})'.format(2*filter_R))

        self.eta_d = eta-self.get_delta_eta_from_length(min_feature_size, self.filter_R)
        self.eta_e = eta+self.get_delta_eta_from_length(min_feature_size, self.filter_R)

        self.g_s_hist = []  #< History of the min feature size penalty term for the "solid"
        self.g_v_hist = []  #< History of the min feature size penalty term for the "void"

        self.beta = beta
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.dz = z[1]-z[0] if (hasattr(z, "__len__") and len(z)>1) else 0
        self.depth = z[-1]-z[0] if (hasattr(z, "__len__") and len(z)>1) else 220e-9
        self.beta_factor = 1.2
        self.discreteness = 0

        self.penalty_scaling_beta_threshold = 99e99 if min_feature_size==0 else 12  #< Very large value (always larger than beta) to disable min feature size constraint
        self.penalty_scaling_factor_max = 1e4
        self.penalty_scaling_factor = min(100*np.square(self.beta - self.penalty_scaling_beta_threshold), self.penalty_scaling_factor_max) if self.beta > self.penalty_scaling_beta_threshold else 0

        ## Prep-work for unfolding symmetry to properly enforce min feature sizes at symmetry boundaries. Off by default for now!
        self.symmetry_x = False
        self.symmetry_y = False
        self.unfold_symmetry = self.symmetry_x or self.symmetry_y #< We do want monitors to unfold symmetry but we need to detect if the size does not match the parameters anymore

    def get_delta_eta_from_length(self, target_length, filter_R):
        scaled_length = target_length/(2*filter_R)
        return np.square(scaled_length) if target_length<filter_R else 0.5-np.square(1-scaled_length)

    def use_interpolation(self):
        return True

    def check_license_requirements(self, sim):
        ## Try to execute one of the topology script commands
        try:
            sim.fdtd.eval(('params = struct;'
                            'params.eps_levels=[1,2];'
                            'params.filter_radius = 3;'
                            'params.beta = 1;'
                            'params.eta = 0.5;'
                            'params.dx = 0.1;'
                            'params.dy = 0.1;'
                            'params.dz = 0.0;'
                            'eps_geo = topoparamstoindex(params,ones(5,5));'))
        except:
            raise UserWarning('Could not execute required topology optimization commands. Either the version of FDTD is outdated or the '
                              'license requirements for topology optimization are not fulfilled. Please contact your support or sales representative.')

        return True

    def calc_penalty_term(self, sim, params):
        if self.penalty_scaling_factor==0:
            return 0

        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.eta_e = {5};'
                       'params.eta_d = {6};'
                       'params.dx = {7};'
                       'params.dy = {8};'
                       'params.dz = 0.0;'
                       'debug;'
                       'min_feature_indicators = topoparamstominfeaturesizeindicator(params,topo_rho);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.eta_e,self.eta_d,self.dx,self.dy) )
        min_feature_indicators = sim.fdtd.getv("min_feature_indicators").flatten()

        self.g_s_hist.append(min_feature_indicators[0])   #< For logging/debugging, not needed for the actual calculation
        self.g_v_hist.append(min_feature_indicators[1])   #< For logging/debugging, not needed for the actual calculation
        
        penalty_fom = -self.penalty_scaling_factor*np.sum(min_feature_indicators)
        return penalty_fom

    def calc_penalty_gradient(self, sim, params):
        if self.penalty_scaling_factor==0:
            return np.zeros_like(params)

        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.eta_e = {5};'
                       'params.eta_d = {6};'
                       'params.dx = {7};'
                       'params.dy = {8};'
                       'params.dz = 0.0;'
                       'debug;'
                       'penalty_grad = topoparamstominfeaturesizegradient(params,topo_rho);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.eta_e,self.eta_d,self.dx,self.dy) )
        penalty_grad = -self.penalty_scaling_factor*sim.fdtd.getv("penalty_grad")
        return penalty_grad.flatten()

    def calc_discreteness(self):
        ''' Computes a measure of discreteness. Is 1 when the structure is completely discrete and less when it is not. '''
        rho = self.calc_params_from_eps(self.eps).flatten()
        return 1 - np.sum(4*rho*(1-rho)) / len(rho)

    def progress_continuation(self):
        self.discreteness = self.calc_discreteness()
        print("Discreteness: {}".format(self.discreteness))

        # If it is sufficiently discrete (99%), we terminate
        if self.discreteness > 0.99:
            return False

        ## Otherwise, we increase beta and keep going
        self.beta *= self.beta_factor
        print('Beta is {}'.format(self.beta))
        
        if self.beta > self.penalty_scaling_beta_threshold:
            self.penalty_scaling_factor = min(100*np.square(self.beta - self.penalty_scaling_beta_threshold), self.penalty_scaling_factor_max)
            print('The penalty scaling factor is {}'.format(self.penalty_scaling_factor))

        return True

    def to_file(self, filename):
        np.savez(filename, params=self.last_params, eps_min=self.eps_min, eps_max=self.eps_max, x=self.x, y=self.y, z=self.z, depth=self.depth, beta=self.beta, eps=self.eps)

    def calc_params_from_eps(self,eps):
        return np.minimum(np.maximum((eps - self.eps_min) / (self.eps_max-self.eps_min),0),1.0)

    def set_params_from_eps(self,eps):
        self.last_params = self.calc_params_from_eps(eps)

    def extract_parameters_from_simulation(self, sim):
        sim.fdtd.selectpartial('import')
        sim.fdtd.eval('set("enabled",0);')

        sim.fdtd.selectpartial('initial_guess')
        sim.fdtd.eval('set("enabled",1);')
        eps = get_eps_from_sim(sim.fdtd, unfold_symmetry=False) #< Don't unfold when getting the inital parameters
        sim.fdtd.selectpartial('initial_guess')
        sim.fdtd.eval('set("enabled",0);')

        sim.fdtd.selectpartial('import')
        sim.fdtd.eval('set("enabled",1);')
        reduced_eps = np.real(eps[0])

        self.set_params_from_eps(reduced_eps)


    def get_eps_from_params(self, sim, params):
        rho = np.reshape(params, (len(self.x),len(self.y)))
        self.last_params = rho

        ## Expand symmetry (either along x- or y-direction)
        rho = self.unfold_symmetry_if_applicable(rho)

        ## Extend boundary to include effects from fixed structure

        ## Use script function to convert the raw parameters to a permittivity distribution and get the result
        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'debug;'
                       'eps_geo = topoparamstoindex(params,topo_rho);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy) )
        eps = sim.fdtd.getv("eps_geo")

        ## Reduce symmetry again (move to cad eventually?)
        if self.symmetry_x:
            shape = rho.shape
            eps = eps[int((shape[0]-1)/2):,:] #np.vstack( (np.flipud(rho)[:-1,:],rho) )
        if self.symmetry_y:
            shape = rho.shape
            eps = eps[:,int((shape[1]-1)/2):] #np.hstack( (np.fliplr(rho)[:,:-1],rho) )

        return eps

    def initialize(self, wavelengths, opt):
        self.opt=opt
        pass

    def update_geometry(self, params, sim):
        self.eps = self.get_eps_from_params(sim, params)
        self.discreteness = self.calc_discreteness()

    def unfold_symmetry_if_applicable(self, rho):
        ## Expand symmetry (either along x- or y-direction)
        if self.symmetry_x:
            rho = np.vstack( (np.flipud(rho)[:-1,:],rho) )
        if self.symmetry_y:
            rho = np.hstack( (np.fliplr(rho)[:,:-1],rho) )
        return rho


    def get_current_params_inshape(self, unfold_symmetry=False):
        return self.last_params

    def get_current_params(self):
        params = self.get_current_params_inshape()
        return np.reshape(params,(-1)) if params is not None else None

    def plot(self,ax_eps):
        ax_eps.clear()
        x = self.x
        y = self.y
        eps = self.eps
        ax_eps.imshow(np.real(np.transpose(eps)), vmin=self.eps_min, vmax=self.eps_max, extent=[min(x)*1e6,max(x)*1e6,min(y)*1e6,max(y)*1e6], origin='lower')

        ax_eps.set_title('Eps')
        ax_eps.set_xlabel('x(um)')
        ax_eps.set_ylabel('y(um)')
        return True

    def write_status(self, f):
        f.write(', {:.4f}, {:.4f}'.format(self.beta, self.discreteness))
        
        if len(self.g_s_hist) > 0:
            f.write(', {:.4g}'.format(self.g_s_hist[-1]))
        else:
            f.write(', {:.4g}'.format(0.0))
        
        if len(self.g_v_hist) > 0:
            f.write(', {:.4g}'.format(self.g_v_hist[-1]))
        else:
            f.write(', {:.4g}'.format(0.0))
class TopologyOptimization2D(TopologyOptimization2DParameters):
    '''
    '''
    self_update = False

    def __init__(self, params, eps_min, eps_max, x, y, z=0, filter_R=200e-9, eta=0.5, beta=1, eps=None, min_feature_size = 0):
        super().__init__(params, eps_min, eps_max, x, y, z, filter_R, eta, beta, eps, min_feature_size = min_feature_size)

    @classmethod
    def from_file(cls, filename, z=0, filter_R=200e-9, eta=0.5, beta = None):
        data = np.load(filename)
        if beta is None:
            beta = data["beta"]
        return cls(data["params"], data["eps_min"], data["eps_max"], data["x"], data["y"], z = z, filter_R = filter_R, eta=eta, beta=beta, eps=data["eps"])

    def set_params_from_eps(self,eps):
        # Use the permittivity in z-direction. Does not really matter since this is just used for the initial guess and is (usually) heavily smoothed
        super().set_params_from_eps(eps[:,:,0,0,2])


    def calculate_gradients_on_cad(self, sim, forward_fields, adjoint_fields, wl_scaling_factor):
        lumapi.putMatrix(sim.fdtd.handle, "wl_scaling_factor", wl_scaling_factor)

        sim.fdtd.eval("V_cell = {};".format(self.dx*self.dy) +
                      "debug;"+
                      "dF_dEps = pinch(sum(2.0 * V_cell * eps0 * {0}.E.E * {1}.E.E,5),3);".format(forward_fields, adjoint_fields) +
                      "num_wl_pts = length({0}.E.lambda);".format(forward_fields) +
                      
                      "for(wl_idx = [1:num_wl_pts]){" +
                      "    dF_dEps(:,:,wl_idx) = dF_dEps(:,:,wl_idx) * wl_scaling_factor(wl_idx);" +
                      "}" + 
                      "dF_dEps = real(dF_dEps);")

        rho = self.get_current_params_inshape()

        ## Expand symmetry (either along x- or y-direction)
        rho = self.unfold_symmetry_if_applicable(rho)

        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'debug;'
                       'dF_dp = topoparamstogradient(params,topo_rho,dF_dEps);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy) )
        # topo_grad = sim.fdtd.getv("dF_dp")

        ## Fold symmetry again (on the CAD this time)
        if self.symmetry_x:
            ## To be tested!
            sim.fdtd.eval(('dF_dp = dF_dp({0}:end,:,:);'
                           'dF_dp(2:end,:,:) = 2*dF_dp(:,2:end,:);').format(int((shape[0]+1)/2)))
        if self.symmetry_y:
            shape = rho.shape
            sim.fdtd.eval(('dF_dp = dF_dp(:,{0}:end,:);'
                           'dF_dp(:,2:end,:) = 2*dF_dp(:,2:end,:);').format(int((shape[1]+1)/2)))

        return "dF_dp"


    def calculate_gradients(self, gradient_fields, sim):

        rho = self.get_current_params_inshape()

        # If we have frequency data (3rd dim), we need to adjust the dimensions of epsilon for broadcasting to work
        E_forward_dot_E_adjoint = np.atleast_3d(np.real(np.squeeze(np.sum(gradient_fields.get_field_product_E_forward_adjoint(),axis=-1))))

        dF_dEps = 2*self.dx*self.dy*eps0*E_forward_dot_E_adjoint
        
        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.putv("dF_dEps", dF_dEps)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'debug;'
                       'topo_grad = topoparamstogradient(params,topo_rho,dF_dEps);'
                       ).format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy))
                        
        topo_grad = sim.fdtd.getv("topo_grad")

        return topo_grad.reshape(-1, topo_grad.shape[-1])


    def add_geo(self, sim, params=None, only_update = False):

        fdtd=sim.fdtd

        eps = self.eps if params is None else self.get_eps_from_params(sim, params.reshape(-1))

        fdtd.putv('x_geo',self.x)
        fdtd.putv('y_geo',self.y)
        fdtd.putv('z_geo',np.array([self.z-self.depth/2,self.z+self.depth/2]))

        if not only_update:
            set_spatial_interp(sim.fdtd,'opt_fields','specified position') 
            set_spatial_interp(sim.fdtd,'opt_fields_index','specified position') 

            script=('select("opt_fields");'
                    'set("x min",{});'
                    'set("x max",{});'
                    'set("y min",{});'
                    'set("y max",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y))
            fdtd.eval(script)

            script=('select("opt_fields_index");'
                    'set("x min",{});'
                    'set("x max",{});'
                    'set("y min",{});'
                    'set("y max",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y))
            fdtd.eval(script)

            script=('addimport;'
                    'set("detail",1);')
            fdtd.eval(script)

            mesh_script=('addmesh;'
                        'set("x min",{});'
                        'set("x max",{});'
                        'set("y min",{});'
                        'set("y max",{});'
                        'set("dx",{});'
                        'set("dy",{});'
                        'debug;').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y),self.dx,self.dy)
            fdtd.eval(mesh_script)

        if eps is not None:
            fdtd.putv('eps_geo',eps)

            ## We delete and re-add the import to avoid a warning
            script=('select("import");'
                    'delete;'
                    'addimport;'
                    'temp=zeros(length(x_geo),length(y_geo),2);'
                    'temp(:,:,1)=eps_geo;'
                    'temp(:,:,2)=eps_geo;'
                    'importnk2(sqrt(temp),x_geo,y_geo,z_geo);')
            fdtd.eval(script)

######## RUNS TOPOLOGY OPTIMIZATION OF A 2D STRUCTURE ########
def runSim(params, eps_min, eps_max, x_pos, y_pos, filter_R):

    ######## DEFINE A 2D TOPOLOGY OPTIMIZATION REGION ########
    geometry = TopologyOptimization2D(params=params, eps_min=eps_min, eps_max=eps_max, x=x_pos, y=y_pos, min_feature_size=28*1e-6, filter_R=filter_R)

    ######## DEFINE FIGURE OF MERIT ########
    # The base simulation script defines a field monitor named 'fom' at the point where we want to modematch to the fundamental TE mode
    fom = ModeMatch(monitor_name = 'fom', mode_number = 'Fundamental TE mode', direction = 'Forward', target_T_fwd=lambda wl: np.ones(wl.size), norm_p = 2)

    ######## DEFINE OPTIMIZATION ALGORITHM ########
    #optimizer = ScipyOptimizers(max_iter=50, method='L-BFGS-B', scaling_factor=1, pgtol=1e-6, ftol=1e-4, target_fom=0.5, scale_initial_gradient_to=0.25)
    optimizer = ScipyOptimizers(max_iter=50, method='L-BFGS-B', scaling_factor=1, pgtol=1e-4, ftol=1e-5, scale_initial_gradient_to=0.0)
    
    #wavelengths = Wavelengths(start = 0.0002997925 , stop = 0.0002997925, points = 1)
    opt = Optimization(base_script=base_geom, wavelengths = 0.0002997925, fom=fom, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=False, plot_history=False, store_all_simulations=True)

    ######## RUN THE OPTIMIZER ########
    opt.run()
    
    # EXTRACTING THE CONTOURS
    


# GEOMETRY
def base_geom(fdtd):
    px_size = 2 #micron
    # SIM PARAMS
    x_points=int(280/px_size)+1
    x_pos = np.linspace(-280/2,280/2,x_points)*1e-6
    
    opt_size_x = max(x_pos)-min(x_pos)
    opt_size_y = 280e-6
     
    size_x = opt_size_x+130e-6
    size_y = opt_size_y+130e-6

    out_wg_dist = 120e-6
    wg_width = 50e-6
    mode_width = 3 * wg_width

    wg_index = round(math.sqrt(11.9),2)
    bg_index = 1

    dx = 0.5e-5

    # INPUT WAVEGUIDE
    fdtd.addrect()
    fdtd.set('name', 'input wg')
    fdtd.set('x min', -size_x)
    fdtd.set('x max', -opt_size_x / 2 + 20e-6)
    fdtd.set('y', 0)
    fdtd.set('y span', wg_width)
    fdtd.set('z', 0)
    fdtd.set('z span', 220e-7)
    fdtd.set('material', '<Object defined dielectric>')
    fdtd.set('index', wg_index)

    # OUTPUT WAVEGUIDES
    fdtd.addrect()
    fdtd.set('name', 'output wg top')
    fdtd.set('x min', opt_size_x / 2 - 20e-6)
    fdtd.set('x max', size_x)
    fdtd.set('y', out_wg_dist - 20e-6)
    fdtd.set('y span', wg_width)
    fdtd.set('z', 0)
    fdtd.set('z span', 220e-7)
    fdtd.set('material', '<Object defined dielectric>')
    fdtd.set('index', wg_index)

    fdtd.addrect()
    fdtd.set('name', 'output wg bottom')
    fdtd.set('x min', opt_size_x / 2 - 20e-6)
    fdtd.set('x max', size_x)
    fdtd.set('y', -out_wg_dist + 20e-6)
    fdtd.set('y span', wg_width)
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
    #fdtd.set('y span', wg_width)
    fdtd.set('z', 0)
    fdtd.set('z span', 1e-6)
    fdtd.set('center wavelength', 0.0002997925)
    fdtd.set('wavelength span', 0)
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
    #fdtd.set('y min bc', 'anti-symmetric')
    #fdtd.set('force symmetric y mesh', 1)
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
    fdtd.set('name', 'fom')
    fdtd.set('monitor type', '2D X-normal')
    fdtd.set('x', opt_size_x / 2 + 10e-6)
    fdtd.set('y', out_wg_dist- 20e-6)
    fdtd.set('y span', opt_size_y/2)

    fdtd.addmesh()
    fdtd.set('name', 'fom_mesh')
    fdtd.set('override x mesh', True)
    fdtd.set('dx', dx)
    fdtd.set('override y mesh', False)
    fdtd.set('override z mesh', False)
    fdtd.set('x', opt_size_x / 2 + 10e-6)
    fdtd.set('x span', 5e-6)
    fdtd.set('y', out_wg_dist- 20e-6)
    fdtd.set('y span', opt_size_y/2)

    # For visualization later
    fdtd.addindex()
    fdtd.set('name', 'global_index')
    fdtd.set('x min', -size_x / 2)
    fdtd.set('x max', size_x / 2)
    fdtd.set('y min', -size_y / 2)
    fdtd.set('y max', size_y / 2)

if __name__ == '__main__':
    size_x = 280		#< Length of the device (in nm). Longer devices typically lead to better performance
    delta_x = 2		#< Size of a pixel along x-axis (in nm)

    size_y = 280		#< Since we use symmetry, this is only have the extent along the y-axis (in nm
    delta_y = 2	#< Size of a pixel along y-axis (in nm)
	
    filter_R = 28	#< Radius of the smoothing filter which removes small features and sharp corners (in nm)
   
    eps_max = 11.9	#< Effective permittivity for a Silicon waveguide with a thickness of 220nm
    eps_min = 1	#< Permittivity of the SiO2 cladding

    x_points=int(size_x/delta_x)+1
    y_points=int(size_y/delta_y)+1

    x_pos = np.linspace(-size_x/2,size_x/2,x_points)*1e-6
    y_pos = np.linspace(0,size_y,y_points)*1e-6

    ## Set initial conditions
    #initial_cond = None                                #< Use the structure 'initial_guess' as defined in the project file

    ## Alternative initial conditions
    #initial_cond = np.ones((x_points,y_points))       #< Start with the domain filled with eps_max
    initial_cond = 0.5*np.ones((x_points,y_points))   #< Start with the domain filled with (eps_max+eps_min)/2
    #initial_cond = np.zeros((x_points,y_points))      #< Start with the domain filled with eps_min         
    runSim(initial_cond, eps_min, eps_max, x_pos, y_pos, filter_R*1e-6)
    
    # debugging script to view the constructed scene before optimization. 

    #with lumapi.FDTD(hide=False) as fdtd:
        ## help(fdtd.addvarfdtd)
        #base_geom(fdtd)
        ## just enough time to take a look around
        #time.sleep(300)