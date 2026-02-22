import math

import torch
import numpy as np
import shapely

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from chspy import CubicHermiteSpline, interpolate_diff_vec
from helpers.helper_functions import get_elevation


def calculate_segment_length(prev_pos, prev_slope, curr_pos, curr_slope, sample_points=100):
    """Calculate length of a spline segment between two anchors."""
    # Create spline once for the segment
    temp_spline = CubicHermiteSpline(n=2)
    temp_spline.add((0, prev_pos, prev_slope))
    temp_spline.add((1, curr_pos, curr_slope))
    
    # Get points and calculate length
    times = np.linspace(0, 1, sample_points)
    spline_points = temp_spline.get_state(times)
    segments = np.diff(spline_points, axis=0)
    segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
    
    return np.sum(segment_lengths)


def calculate_spline_with_distances(anchors):
    """Calculate cumulative distances and create normalized spline in one pass."""
    if not anchors:
        raise ValueError("No anchor points provided")
        
    distances = [0]  # Start with 0
    normalized_spline = CubicHermiteSpline(n=2)
    normalized_spline.add((0, anchors[0][1], anchors[0][2]))  # Add first anchor
    
    # Calculate distances and build normalized spline simultaneously
    for i in range(1, len(anchors)):
        prev_pos = anchors[i-1][1][:2]     # get x,y of previous anchor
        prev_slope = anchors[i-1][2][:2]   # get dx,dy of previous anchor
        curr_pos = anchors[i][1][:2]       # get x,y of current anchor
        curr_slope = anchors[i][2][:2]     # get dx,dy of current anchor
        
        dist = calculate_segment_length(prev_pos, prev_slope, curr_pos, curr_slope)
        distances.append(distances[-1] + dist)
        normalized_spline.add((distances[-1], anchors[i][1], anchors[i][2]))
    
    return normalized_spline, distances


def get_spline_points_and_derivatives(spline, dx=100.0):
    """Get positions and derivatives for multiple points along the spline."""
    # Get anchors and build normalized spline in one pass
    anchors = spline[:]
    if not anchors:
        raise ValueError("Spline has no anchor points")
        
    normalized_spline, distances = calculate_spline_with_distances(anchors)

    # Generate sample points at regular intervals
    total_length = distances[-1]
    ds = np.arange(0, total_length, dx)
    
    # Get positions at each sample point
    positions = normalized_spline.get_state(ds)
    
    # Get derivatives for each point (using vectorized operations where possible)
    derivatives = np.array([
        interpolate_diff_vec(d, normalized_spline.get_anchors(d))
        for d in ds
    ])
        
    return np.hstack([positions, derivatives]), total_length


def convert_angles_to_slopes(thetas, lengths):
    """Convert angles and lengths to x,y slopes."""
    slopes = np.zeros((len(thetas), 2))
    slopes[:, 0] = np.cos(thetas) * lengths
    slopes[:, 1] = np.sin(thetas) * lengths
    return slopes


def solution2spline(solution, design_space):
    """Convert a solution vector to a CubicHermiteSpline."""
    if len(solution) % 3 != 0:
        raise ValueError(f"Solution length {len(solution)} is not divisible by 3")
        
    n_points = len(solution) // 3
    
    solution_indices = solution[:n_points].astype(int)
    solution_thetas = solution[n_points:2*n_points].astype(float)
    solution_length = solution[2*n_points:].astype(float)
    
    # Check for valid indices
    if np.any(solution_indices < 0) or np.any(solution_indices >= len(design_space)):
        raise ValueError(f"Invalid indices in solution: min={solution_indices.min()}, "
                        f"max={solution_indices.max()}, design_space={len(design_space)}")
    
    solution_slopes = convert_angles_to_slopes(solution_thetas, solution_length)
    design_hor_coords = design_space[solution_indices, :2]
    
    # Create spline only once with all points
    spline = CubicHermiteSpline(n=2)
    for i, (coor, slope) in enumerate(zip(design_hor_coords, solution_slopes)):
        spline.add((i, coor, slope))
        
    return spline


class Solution2Cable_design:
    """Converts solution vectors to cable designs with elevation data."""
    
    def __init__(self, design_space, topo_data, cable_spacing=100.0):
        """
        Initialize the converter.
        
        Args:
            design_space: Array of possible coordinate points
            topo_data: Topography data for elevation lookup
            cable_spacing: Distance between points in the generated cable design
        """
        self.design_space = design_space
        self.topo_data = topo_data
        self.cable_spacing = cable_spacing
        
    def __call__(self, solution):
        """
        Convert a solution to a cable design.
        
        Returns:
            tuple: (torch.Tensor of design points and derivatives, cable length)
                  or (None, length) if any elevations are invalid
        """
        try:
            # Create spline from solution
            orig_spline = solution2spline(solution, self.design_space)
            
            # Get points and derivatives along the spline
            points_and_derivatives, length = get_spline_points_and_derivatives(
                orig_spline, dx=self.cable_spacing)
            
            # Extract coordinates and derivatives
            coordinates = points_and_derivatives[:, :2]
            derivatives = points_and_derivatives[:, 2:]
            
            # Create design array [x, y, z, dx, dy, dz]
            design = self._create_design_array(coordinates, derivatives)
            
            if design is None:
                return None, length
                
            return torch.from_numpy(design).float(), length
            
        except Exception as e:
            print(f"Error converting solution to cable design: {e}")
            return None, 0.0
    
    def _create_design_array(self, coordinates, derivatives):
        """Create the design array with elevations and normalized derivatives."""
        design = np.zeros((len(coordinates), 6))
        design[:, :2] = coordinates
        
        # Get elevations at coordinates in one batch
        design[:, 2] = get_elevation(coordinates, self.topo_data).numpy()
        
        # Check for invalid elevations
        if np.isnan(design[:, 2]).any():
            return None
        
        # Calculate elevation gradient
        grad_x = derivatives[:, 0]
        grad_y = derivatives[:, 1]
        
        shifted_coord = coordinates + np.stack([grad_x, grad_y], axis=-1)
        
        # Get elevations for gradient calculation in one batch
        shifted_elevations = get_elevation(shifted_coord, self.topo_data).numpy()
        grad_z = shifted_elevations - design[:, 2]
        
        if np.isnan(grad_z).any():
            return None
        
        # Store derivatives
        design[:, 3:5] = derivatives  # Use slice assignment for better performance
        design[:, 5] = grad_z
        
        # Normalize derivatives
        norms = np.linalg.norm(design[:, 3:], axis=1)
        # Avoid division by zero
        valid_indices = norms > 1e-10
        if not np.all(valid_indices):
            return None
            
        design[valid_indices, 3:] /= norms[valid_indices, np.newaxis]
        
        return design
    
    
class LengthPenalty:
    """Penalty function for cable length deviation from target."""
    
    def __init__(self, target_length, scale=0.1, acceptable_difference=100):
        """
        Initialize the length penalty calculator.
        
        Args:
            target_length: The ideal cable length
            scale: Scaling factor for the penalty
            acceptable_difference: Allowable deviation before applying penalty
        """
        self.target_length = target_length
        self.scale = scale
        self.acceptable_difference = acceptable_difference
        
    def __call__(self, length):
        """
        Calculate penalty for cable length deviation.
        
        Args:
            length: Actual cable length
            
        Returns:
            tuple: (penalty_value, cutoff_flag)
                    penalty_value is negative for lengths exceeding target
                    cutoff_flag is True if length exceeds acceptable threshold
        """
        diff = length - self.target_length
        
        # Determine if length exceeds maximum acceptable threshold
        cutoff = diff > 2 * self.acceptable_difference
        
        # Apply penalty only for cables longer than target
        if diff > 0:
            # Quadratic penalty scaled by acceptable difference
            penalty = -self.scale * (diff / self.acceptable_difference)**2
        else:
            penalty = 0.0
            
        return penalty, cutoff
        
        
class ProgressBar:
    def __init__(self, max_generations):
        self.pbar = tqdm(total=max_generations, desc='GA')
        self.pbar.set_postfix({'Fitness': 0.0,
                               'Discard': '0.0 | 0.0 | 0.0'
                                 })
        
        self.generation = 0
    
    def callback(self, ga_instance):
        self.pbar.update(1)
        self.pbar.set_postfix({'Fitness': ga_instance.best_solution()[1],
                               'Discard': f'{ga_instance.num_wrong_slopes/ga_instance.total_eval:.4f} | {ga_instance.num_wrong_areas/ga_instance.total_eval:.4f} | {ga_instance.num_wrong_lengths/ga_instance.total_eval:.4f}  '
                                 })
        self.generation += 1
        if self.generation >= ga_instance.num_generations:
            self.pbar.close()
            
class ProgressPlotter(ProgressBar):
    def __init__(self, max_generations, solution2cable_design, design_space, topo_data, area_constraint=None):
        super().__init__(max_generations)
        self.solution2cable_design = solution2cable_design
        self.design_space = design_space
        self.topo_data = topo_data
        self.plot_interval = 10
        self.plot_count = 0
        self.area_constraint = area_constraint
        
    def callback(self, ga_instance):
        super().callback(ga_instance)
        
        # Plot every N generations or on the final generation
        if self.generation % self.plot_interval == 0 or self.generation >= ga_instance.num_generations - 1:
            self.plot_count += 1
            solution = ga_instance.best_solution()[0]
            fitness = ga_instance.best_solution()[1]
            
            cable_spline = solution2spline(solution, self.design_space)
            design, cable_length = self.solution2cable_design(solution)
            
            full_cable = CubicHermiteSpline(n=3)
            for i, row in enumerate(design):
                full_cable.add((i, row[:3], row[3:6]*1e2))
                
            times = np.linspace(0,len(full_cable)-1,100)
            design_cable_values = full_cable.get_state(times)
            
            fig, ax = plt.subplots(figsize=(4, 4))
            
            plot_das_design(ax, design_cable_values, design, self.design_space, self.topo_data, 
                            cable_spline, ga_instance, self.generation, fitness, cable_length, self.area_constraint)
            
            plt.tight_layout()
            plt.show()

def plot_das_design(ax, design_cable_values, design, design_space, topo_data, cable_spline, 
                    ga_instance, generation, fitness, cable_length, area_constraint=None):
    """
    Plot DAS cable design and constraints on given axes.
    
    Args:
        ax: Matplotlib axes to plot on
        design_cable_values: Points along the cable spline
        design: DAS cable design points
        design_space: Available design space points
        topo_data: Topography data
        cable_spline: Spline representation of the cable
        ga_instance: Genetic algorithm instance
        generation: Current generation number
        fitness: Current best fitness
        cable_length: Length of the cable design
    """
    # Plot terrain
    ax.imshow(
        topo_data.T,
        extent=[topo_data.easting.min(), topo_data.easting.max(),
                topo_data.northing.min(), topo_data.northing.max()],
        origin='lower', cmap='Greys', zorder=-2)
    
    # Plot all design space points
    ax.scatter(
        design_space[:, 0], design_space[:, 1],
        facecolor='lightblue', linewidths=0.0,
        s=5, label='Design space', alpha=0.3)
    
    # Plot spline
    ax.plot(
        design_cable_values[:, 0], design_cable_values[:, 1], alpha=0.5,
        color='tab:red', label='Cable')
    
    # Add arrows for orientation
    optimal_length = 100
    # Plot original gradients at anchor points
    for i, coords, slopes in cable_spline:
        ax.quiver(
            coords[0], coords[1],  # position
            slopes[0]*optimal_length, slopes[1]*optimal_length,  # direction
            color='black', pivot='middle', width=0.01,
            scale=1, headaxislength=0, headlength=0, headwidth=0,
            scale_units='xy', zorder=2, alpha=0.8)
    
    ax.quiver(
        design[:, 0], design[:, 1],
        design[:, 3]*optimal_length/2, design[:, 4]*optimal_length/2,
        pivot='middle', color='darkred', width=0.005, alpha=0.7,
        scale=1, headaxislength=0, headlength=0, headwidth=0,
        scale_units='xy', zorder=2)
    
    
    # Add area constraint if it exists
    if area_constraint is not None:
        ax.plot(
            area_constraint.exterior.coords.xy[0],
            area_constraint.exterior.coords.xy[1],
            color='k', linewidth=1.0, linestyle='--', zorder=20, alpha=0.5,
            label='Constraint')
    
    ax.set_aspect('equal')
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    title = f"Generation {generation}/{ga_instance.num_generations}\n"
    title += f"EIG: {fitness:.2f}, Length: {cable_length:.1f}m"
    ax.set_title(title)
    
    # Set limits to design space + a bit of margin
    margin = 200
    ax.set_xlim(design_space[:, 0].min()-margin, design_space[:, 0].max()+margin)
    ax.set_ylim(design_space[:, 1].min()-margin, design_space[:, 1].max()+margin)
    
class DASFitnessFunction:
    def __init__(self, penalty_func, area_constraint, solution2cable_design, BED_class, NMC_kwargs
    ):
        self.penalty_func = penalty_func
        self.area_constraint = area_constraint
        self.solution2cable_design = solution2cable_design
        self.BED_class = BED_class
        self.NMC_kwargs = NMC_kwargs
        
    def __call__(self, ga_instance, solution, solution_idx):
        ga_instance.total_eval += 1

        # Convert solution to design
        design, length = self.solution2cable_design(solution)
        if design is None:
            ga_instance.num_wrong_slopes += 1
            return -math.inf

        # Check if design points are within area constraint
        design_coords = design[:, :2]
        points_inside = [self.area_constraint.contains(shapely.Point(coord)) for coord in design_coords]
        if not all(points_inside):
            ga_instance.num_wrong_areas += 1
            return -math.inf

        # Check length constraint
        penalty, cut_off = self.penalty_func(length)
        if cut_off:
            ga_instance.num_wrong_lengths += 1
            return -math.inf 

        # Calculate expected information gain
        eig = self.BED_class.calculate_EIG(
            design,
            eig_method='NMC',
            eig_method_kwargs=self.NMC_kwargs,
            random_seed=0,
            progress_bar=False,
        )[0].item()

        return eig + penalty