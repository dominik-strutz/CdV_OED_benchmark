import pandas as pd
import xarray as xr
import torch

from .helper_functions import (
    convex_hull2D_prior_dist_constructor,
    concave_hull2D_prior_dist_constructor
)

MIN_EASTING = 2701200.0
MAX_EASTING = 2703700.0

MIN_NORTHING = 1171500.0
MAX_NORTHING = 1174500.0

topo_data = xr.load_dataarray('data/geographic_data/topo_data.nc')

nodes_full = pd.read_csv('data/geographic_data/nodes_full.csv')
nodes_shoulder = pd.read_csv('data/geographic_data/nodes_shoulder.csv')
das_full = pd.read_csv('data/geographic_data/das_full.csv')

design_space_full = torch.tensor(
    nodes_full[['easting', 'northing', 'elevation']].values,
    dtype=torch.float32
)
design_space_shoulder = torch.tensor(
    nodes_shoulder[['easting', 'northing', 'elevation']].values,
    dtype=torch.float32
)

indices_full = torch.arange(design_space_full.shape[0])
indices_shoulder = torch.arange(design_space_shoulder.shape[0])

design_space_shoulder = design_space_shoulder[indices_shoulder]

design_space_full = torch.hstack([design_space_full, indices_full.reshape(-1, 1)])
design_space_shoulder = torch.hstack([design_space_shoulder, indices_shoulder.reshape(-1, 1)])
    
picking_stats_lines = pd.read_csv('data/picking_stats/picking_stats_lines.csv')
picking_stats_shoulder = pd.read_csv('data/picking_stats/picking_stats_shoulder.csv')

events_full = torch.tensor(
    picking_stats_lines[['src_easting', 'src_northing', 'src_elevation']].drop_duplicates().values,
    dtype=torch.float32
)
events_shoulder = torch.tensor(
    picking_stats_shoulder[['src_easting', 'src_northing', 'src_elevation']].drop_duplicates().values,
    dtype=torch.float32
)

indices_lines = torch.arange(events_full.shape[0])
indices_shoulder = torch.arange(events_shoulder.shape[0])

events_full = torch.hstack([events_full, indices_lines.reshape(-1, 1)])
events_shoulder = torch.hstack([events_shoulder, indices_shoulder.reshape(-1, 1)])


X, Y = torch.meshgrid(
    torch.from_numpy(topo_data['easting'].values).float(),
    torch.from_numpy(topo_data['northing'].values).float(),
    indexing='ij'
    )

grid_coords_hor  = torch.vstack([X.ravel(), Y.ravel()]).T
grid_coords_vert = torch.from_numpy(topo_data.values.ravel()).float()

grid_coords = torch.hstack([grid_coords_hor, grid_coords_vert.reshape(-1, 1)])

convex_hull_shoulder = convex_hull2D_prior_dist_constructor(
    design_space_shoulder[:, :2].numpy(),
    topo_data,
    buffer=20,
)

concave_hull_full = concave_hull2D_prior_dist_constructor(
    design_space_full[:835, :2].numpy(),
    topo_data,
    buffer=20,
    ratio=0.05,
)
