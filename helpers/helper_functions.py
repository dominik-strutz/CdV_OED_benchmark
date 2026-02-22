import ast
import numpy as np

import torch
from torch import Tensor, Size
import torch.distributions as dist
import zuko

import xarray as xr

import shapely
from matplotlib.path import Path

savfig_kwargs = dict(
    dpi=1000,
    bbox_inches='tight',
    facecolor=None,
    pad_inches=0.1)

#################################################################################
################################# get_elevation #################################
#################################################################################

def get_elevation(points, topo_data):
    east = xr.DataArray(points[..., 0], dims='points')
    north = xr.DataArray(points[..., 1], dims='points')
    elevations = topo_data.interp(
        easting=east, northing=north, method='linear').values
    return torch.from_numpy(elevations)

#################################################################################
############################ ConvexHull_Distribution ############################
#################################################################################

class Sobol_BoxUniform(zuko.distributions.BoxUniform):
    def __init__(self, lower, upper, ndims: int = 1):
        super().__init__(lower, upper, ndims)
        self.low = lower if torch.is_tensor(lower) else torch.tensor(lower)
        self.high = upper if torch.is_tensor(upper) else torch.tensor(upper)
    def rsample(self, sample_shape=torch.Size()):
        n = torch.prod(torch.tensor(sample_shape))
        output_shape = sample_shape + self.low.shape
        unscaled_samples =  torch.quasirandom.SobolEngine(
            dimension=self.low.shape[0],
            scramble=True,
            seed=torch.randint(0, 1000000, (1,)).item() ).draw(n).to(self.low.device).reshape(output_shape)
        return self.low + unscaled_samples * (self.high - self.low)
    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)


class Hull_Distribution(dist.Distribution):
    def __init__(self, points, topo_data, depth=200, base_dist='uniform'):
        
        if points.shape[1] != 2:
            raise ValueError("Points must be 2-dimensional. Probably extendable to n-dimensional.")
        
        self.points = points        
        self.shapley_poly = shapely.geometry.Polygon(points)
        
        self.poly_area = self.shapley_poly.area
        
        if depth > 0:
            self.poly_volume = self.poly_area * depth
        else:
            self.poly_volume = self.poly_area * 1.0
        
        self.const_prob = 1.0 / self.poly_volume
        self.log_const_prob = np.log(self.const_prob)
                
        self.bounds = self.shapley_poly.bounds
        self.bounds_area = (self.bounds[2]-self.bounds[0])*(self.bounds[3]-self.bounds[1])
        
        if base_dist == 'uniform':
            self.box_dist = zuko.distributions.BoxUniform(
                lower=self.bounds[:2], upper=self.bounds[2:])
        elif base_dist == 'sobol':
            self.box_dist = Sobol_BoxUniform(
                lower=self.bounds[:2], upper=self.bounds[2:])

        else:
            raise ValueError(f'base_dist must be "uniform" or "sobol", not {base_dist}')

        # self.convex_hull = ConvexHull(self.points)
        self.hull_path = Path(np.array(self.shapley_poly.exterior.xy).T)
        
        self.topo_data = topo_data
        self.depth = depth
        self.vert_dist = dist.Uniform(-1, self.depth)
        
    def sample(self, sample_shape: Size = 1) -> Tensor:
        
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        if isinstance(sample_shape, tuple):
            sample_shape = torch.Size(sample_shape)
        if isinstance(sample_shape, list):
            sample_shape = torch.Size(sample_shape)
        

        n = torch.prod(torch.tensor(sample_shape))
        n_buffer = int(n*(self.bounds_area/self.shapley_poly.area)*2.0)

        while True:
            rand_points = self.box_dist.sample((n_buffer,))
                    
            rand_points = rand_points[
                self.hull_path.contains_points(rand_points)
            ]
            
            if len(rand_points) >= n:
                break
        
        rand_points = rand_points[:n]

        elevations = get_elevation(rand_points, self.topo_data)
        depth = torch.rand(n) * self.depth
        rand_points = torch.cat(
            (rand_points, (elevations-depth)[:, None]), dim=1)        
        rand_points = rand_points.reshape(sample_shape + (3,))
            
        return rand_points
    
    
    def log_prob(self, value: Tensor, fast_eval=True) -> Tensor:
        
        if fast_eval:
            return torch.full_like(value[..., 0], self.log_const_prob)
        else:
            if value.ndim == 1:
                value = value[None, ...]
            elevation = get_elevation(value[..., :2].detach(), self.topo_data)
            log_prob = torch.full_like(value[..., 0], self.log_const_prob)
            
            mask_hor = [not self.shapley_poly.contains(shapely.geometry.Point(p)) \
                for p in value[..., :2]]
            
            log_prob[mask_hor] = torch.tensor(-np.inf)
            
            # mask everythong not within between 0 and depth
            mask_vert = (elevation - value[..., 2] < -1) | (elevation - value[..., 2] > self.depth)
            
            log_prob[mask_vert] = torch.tensor(-np.inf)
            
            return log_prob
        

#################################################################################
########################### surface field distribution ##########################
################################################################################# 

class SurfaceField_Distribution(dist.Distribution):
    def __init__(self, distibution, topo_data, depth=200):
        
        self.hor_distibution = distibution
        self.topo_data = topo_data
        self.depth = depth
        
        self.bounds = torch.tensor(
            [[self.topo_data['easting'].values.min()+60, self.topo_data['northing'].values.min()+60],
             [self.topo_data['easting'].values.max()-60, self.topo_data['northing'].values.max()-60]])
        
        self.vert_dist = dist.Uniform(-1, self.depth)
        
        
    def sample(self, sample_shape: Size = 1) -> Tensor:
        
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        if isinstance(sample_shape, tuple):
            sample_shape = torch.Size(sample_shape)
        if isinstance(sample_shape, list):
            sample_shape = torch.Size(sample_shape)
        
        n = torch.prod(torch.tensor(sample_shape))
        rand_points = self.hor_distibution.sample((n,))

        rand_points = torch.clamp(rand_points, self.bounds[0], self.bounds[1])

        elevations = get_elevation(rand_points, self.topo_data)
        depth = torch.rand(n) * self.depth
        rand_points = torch.cat(
            (rand_points, (elevations-depth)[:, None]), dim=1)        
        rand_points = rand_points.reshape(sample_shape + (3,))
        
        offset_from_bounds = 0.01 * (self.bounds[1] - self.bounds[0])
        
        # clip to bounds
        rand_points[..., 0] = torch.clamp(
            rand_points[..., 0], self.topo_data['easting'].values.min()+offset_from_bounds[0],
            self.topo_data['easting'].values.max()-offset_from_bounds[0])
        rand_points[..., 1] = torch.clamp(
            rand_points[..., 1], self.topo_data['northing'].values.min()+offset_from_bounds[1],
            self.topo_data['northing'].values.max() - offset_from_bounds[1])

        return rand_points.float()
    
    def log_prob(self, value: Tensor, fast_eval=True) -> Tensor:
        
        if value.ndim == 1:
            value = value[None, ...]
        log_prob_hori = self.hor_distibution.log_prob(value[..., :2])        

        if not fast_eval:
            elevation = get_elevation(value[..., :2].detach(), self.topo_data)
            
            log_prob_vert = self.vert_dist.log_prob(elevation - value[..., 2])
            # quite slow and we could assume that points are always within bounds
        else:
            log_prob_vert = self.vert_dist.log_prob(0.5 * self.depth)
        
        return log_prob_hori + log_prob_vert


#################################################################################
############################## prior distributions ##############################
#################################################################################

def convex_hull2D_prior_dist_constructor(
    points,
    topography,
    buffer=None,
    depth =None,
    base_dist='uniform'):
    
    # Create a convex hull from the points
    hull = shapely.geometry.MultiPoint(points).convex_hull
    
    if buffer is not None:
        hull = hull.buffer(buffer)
        
    if depth is None:
        depth = 0
        
    return Hull_Distribution(
        np.array(hull.exterior.xy).T,
        topography, depth, base_dist)


def concave_hull2D_prior_dist_constructor(
    points,
    topography,
    buffer=None,
    depth = None,
    ratio=0.1,
    base_dist='uniform'
    ):
    
    # Create a convex hull from the points
    hull = shapely.concave_hull(shapely.geometry.MultiPoint(points), ratio)
    
    if buffer is not None:
        hull = hull.buffer(buffer)
        
    if depth is None:
        depth = 0
        
    return Hull_Distribution(
        np.array(hull.exterior.xy).T,
        topography, depth, base_dist)

#######################################################################################################
# Desin Loading Helpers 
#######################################################################################################

def get_design_information(df, scenario):
    
    study_area = scenario.get('study_area', 'full')
    assert study_area in ['full', 'shoulder']
    
    if study_area == 'full':
        model_prior = scenario.get('model_prior', 'displacement')
    elif study_area == 'shoulder':
        model_prior = scenario.get('model_prior', 'uniform')
    else:
        raise ValueError(f"Invalid study area: {study_area}")
    
    velocity_model = scenario.get('velocity_model', 'gradient')    
    
    if velocity_model == 'heterogeneous':
        vel_sigma = scenario.get('vel_sigma', 0.01)
    elif velocity_model in ['gradient', 'homogeneous']:
        vel_sigma = scenario.get('vel_sigma', 0.05)
    else:
        raise ValueError(f"Invalid velocity model: {velocity_model}")
    assert vel_sigma in [0.005, 0.01, 0.05, 0.1]
    
    noise_correlation = scenario.get('noise_correlation', 100.0)
    assert noise_correlation in [0.0, 50.0, 100.0, 200.0, 400.0]
    
    drop_mean = scenario.get('drop_mean', 0.0)
    assert drop_mean in [0.0, 0.2,  0.35, 0.5]
    
    drop_gradient = scenario.get('drop_gradient', 0.0)
    assert drop_gradient in [0.0, -15.0, -30.0, -100.0]
    
    optimisation = scenario.get('optimisation', 'genetic')
    assert optimisation in ['genetic', 'random', 'sobol', 'iterative']
    
    EIG_method = scenario.get('EIG_method', 'NMC')
    assert EIG_method in ['NMC', 'DN']
    
    EIG_N = scenario.get('EIG_N', 1000)
    assert EIG_N in [int(1e2), int(2e2), int(5e2), int(1e3), int(2e3), int(5e3), int(1e4)]
    
    df = df[
        (df['study_area'] == study_area) &
        (df['model_prior'] == model_prior) &
        (df['velocity_model'] == velocity_model) &
        (df['vel_sigma'] == vel_sigma) &
        (df['noise_correlation'] == noise_correlation) &
        (df['drop_mean'] == drop_mean) &
        (df['drop_gradient'] == drop_gradient) &
        (df['optimisation'] == optimisation) &
        (df['EIG_method'] == EIG_method) &
        (df['EIG_N'] == EIG_N)
    ]
    
    out = {
        'design': [np.array(ast.literal_eval(design)) for design in df['design'].values],
        'N_rec': [int(N_rec) for N_rec in df['N_rec'].values],
        'EIG': [float(EIG) for EIG in df['EIG'].values],
        'EIG_ref': [float(EIG_ref) for EIG_ref in df['EIG_ref'].values],
        'runtime': [float(runtime) for runtime in df['runtime'].values],
        'study_area': study_area,
        'model_prior': model_prior, 
        'velocity_model': velocity_model,
    }
    
    return out