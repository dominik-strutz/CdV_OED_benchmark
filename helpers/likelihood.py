import torch
import torch.distributions as dist


class WeightedMultivariateNormal:
    """
    Thin wrapper around MultivariateNormal that removes weighted mean before evaluating log_prob.
    The weights are derived from the precision matrix (inverse of covariance).
    """
    
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        # Store the original loc for weighted mean calculation
        self._original_loc = loc
        
        # Calculate weights from precision matrix
        if precision_matrix is not None:
            self._precision_matrix = precision_matrix
        elif scale_tril is not None:
            # Calculate precision matrix from scale_tril
            batch_size = scale_tril.shape[0]
            dim = scale_tril.shape[-1]
            identity = torch.eye(dim, device=scale_tril.device).expand(batch_size, dim, dim)
            self._precision_matrix = torch.linalg.solve_triangular(
                scale_tril, 
                torch.linalg.solve_triangular(
                    scale_tril.transpose(-1, -2), 
                    identity,
                    upper=True
                ),
                upper=False
            )
        elif covariance_matrix is not None:
            self._precision_matrix = torch.linalg.inv(covariance_matrix)
        else:
            raise ValueError("Must provide either covariance_matrix, precision_matrix, or scale_tril")
        
        # Calculate weights (sum of precision matrix rows, normalized)
        weights = torch.sum(self._precision_matrix, dim=-1)
        self._weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        
        # Calculate weighted mean and subtract from loc
        weighted_mean = torch.sum(loc * self._weights, dim=-1, keepdim=True)
        adjusted_loc = loc - weighted_mean
        
        # Create the underlying distribution with adjusted loc
        self._dist = dist.MultivariateNormal(adjusted_loc, covariance_matrix=covariance_matrix, 
                                           precision_matrix=precision_matrix, scale_tril=scale_tril, 
                                           validate_args=validate_args)
    
    def log_prob(self, value):
        # Calculate weighted mean of the input value and subtract it
        weighted_mean = torch.sum(value * self._weights, dim=-1, keepdim=True)
        adjusted_value = value - weighted_mean
        return self._dist.log_prob(adjusted_value)
    
    def sample(self, sample_shape=torch.Size()):
        # Sample from the adjusted distribution and add back the weighted mean of original loc
        samples = self._dist.sample(sample_shape)
        original_weighted_mean = torch.sum(self._original_loc * self._weights, dim=-1, keepdim=True)
        return samples + original_weighted_mean
    
    def expand(self, batch_shape, _instance=None):
        # Expand the underlying distribution and create a new wrapper with expanded weights
        expanded_dist = self._dist.expand(batch_shape, _instance)
        
        # Create new instance with expanded parameters
        new_instance = WeightedMultivariateNormal.__new__(WeightedMultivariateNormal)
        new_instance._dist = expanded_dist
        
        # Expand weights and original_loc to match the new batch shape
        expanded_batch_shape = torch.Size(batch_shape)
        original_batch_shape = self._weights.shape[:-1]
        
        # Calculate the expansion dimensions
        expand_dims = expanded_batch_shape + self._weights.shape[len(original_batch_shape):]
        expand_dims_loc = expanded_batch_shape + self._original_loc.shape[len(original_batch_shape):]
        
        new_instance._weights = self._weights.expand(expand_dims)
        new_instance._original_loc = self._original_loc.expand(expand_dims_loc)
        new_instance._precision_matrix = self._precision_matrix.expand(
            expanded_batch_shape + self._precision_matrix.shape[len(original_batch_shape):])
        
        return new_instance
    
    def __getattr__(self, name):
        # Delegate all other attributes/methods to the underlying distribution
        return getattr(self._dist, name)


class WeightedIndependentNormal:
    """
    Thin wrapper around Independent Normal that removes weighted mean before evaluating log_prob.
    The weights are derived from the inverse variance (precision).
    """
    
    def __init__(self, loc, scale, validate_args=None):
        # Store the original loc for weighted mean calculation
        self._original_loc = loc
        
        # Calculate weights from precision (inverse variance squared)
        precision = scale.reciprocal() ** 2
        self._weights = precision / torch.sum(precision, dim=-1, keepdim=True)
        
        # Calculate weighted mean and subtract from loc
        weighted_mean = torch.sum(loc * self._weights, dim=-1, keepdim=True)
        adjusted_loc = loc - weighted_mean
        
        # Create the underlying distribution with adjusted loc
        base_dist = dist.Normal(adjusted_loc, scale, validate_args=validate_args)
        self._dist = dist.Independent(base_dist, 1, validate_args=validate_args)
    
    def log_prob(self, value):
        # Calculate weighted mean of the input value and subtract it
        weighted_mean = torch.sum(value * self._weights, dim=-1, keepdim=True)
        adjusted_value = value - weighted_mean
        return self._dist.log_prob(adjusted_value)
    
    def sample(self, sample_shape=torch.Size()):
        # Sample from the adjusted distribution and add back the weighted mean of original loc
        samples = self._dist.sample(sample_shape)
        original_weighted_mean = torch.sum(self._original_loc * self._weights, dim=-1, keepdim=True)
        return samples + original_weighted_mean
    
    def expand(self, batch_shape, _instance=None):
        # Expand the underlying distribution and create a new wrapper with expanded weights
        expanded_dist = self._dist.expand(batch_shape, _instance)
        
        # Create new instance with expanded parameters
        new_instance = WeightedIndependentNormal.__new__(WeightedIndependentNormal)
        new_instance._dist = expanded_dist
        
        # Expand weights and original_loc to match the new batch shape
        expanded_batch_shape = torch.Size(batch_shape)
        original_batch_shape = self._weights.shape[:-1]
        
        # Calculate the expansion dimensions
        expand_dims = expanded_batch_shape + self._weights.shape[len(original_batch_shape):]
        expand_dims_loc = expanded_batch_shape + self._original_loc.shape[len(original_batch_shape):]
        
        new_instance._weights = self._weights.expand(expand_dims)
        new_instance._original_loc = self._original_loc.expand(expand_dims_loc)
        
        return new_instance
    
    def __getattr__(self, name):
        # Delegate all other attributes/methods to the underlying distribution
        return getattr(self._dist, name)

def decompose_covariance_matrix(diagonal, correlation_matrix):
    """
    Decompose the covariance matrix using the diagonal and normalized correlation matrix for batches.

    Args:
        diagonal (torch.Tensor): A 2D tensor containing the standard deviations (sqrt of variances) for each batch.
        correlation_matrix (torch.Tensor): A 3D tensor containing the correlation matrices for each batch.

    Returns:
        torch.Tensor: The lower triangular scale matrices for each batch.
    """
    # Input validation
    if diagonal.dim() != 2:
        raise ValueError("Diagonal must be a 2D tensor.")
    if correlation_matrix.dim() != 3:
        raise ValueError("Correlation matrix must be a 3D tensor.")
    if diagonal.size(1) != correlation_matrix.size(1) or diagonal.size(1) != correlation_matrix.size(2):
        raise ValueError("Dimensions of diagonal and correlation matrix must match.")

    batch_size, n, _ = correlation_matrix.size()
    
    # Normalize correlation matrix row-wise
    row_sums = correlation_matrix.sum(dim=-1, keepdim=True)
    normalized_correlation = correlation_matrix / row_sums
    
    # Ensure diagonal elements are 1
    diag_mask = torch.eye(n, device=correlation_matrix.device).bool().unsqueeze(0)
    normalized_correlation = normalized_correlation.masked_fill(diag_mask, 1.0)
    
    # Construct lower triangular matrix
    L = torch.zeros_like(normalized_correlation)
    L[..., torch.arange(n), torch.arange(n)] = 1.0
    
    # Fill lower triangular part
    mask = torch.tril(torch.ones(n, n), diagonal=-1).bool().unsqueeze(0)
    L[mask.expand(batch_size, -1, -1)] = normalized_correlation[mask.expand(batch_size, -1, -1)]
    
    # Scale by diagonal
    D = torch.diag_embed(diagonal)
    scale_tril = D @ L

    return scale_tril

class BaseDataLikelihood:
    def __init__(
        self, forward_function, vel_sigma,
        tt_obs_std=0.01, dependence_distance=100.0):
        
        self.forward_function = forward_function
        self.vel_sigma = vel_sigma
        self.tt_obs_std = tt_obs_std
        self.dependence_distance = dependence_distance

    def forward(self, model_samples, design):
        return self.forward_function(model_samples, design)

    def cov(self, tt, model_samples, design):
        tt_std = torch.sqrt(tt) * self.vel_sigma

        if self.dependence_distance > 0:
            distance_matrix = torch.cdist(design, design, p=2.0).double()
            
            # distance_matrix = (1 - distance_matrix / self.dependence_distance).clamp(0.0, 0.7)
            # distance_matrix.diagonal(dim1=-1, dim2=-2).fill_(1.0)
            # correlation_matrix = distance_matrix.unsqueeze(0).expand(model_samples.shape[0], -1, -1
            
            # Apply exponential correlation function instead of linear drop-off
            distance_matrix = torch.exp(-0.5 * distance_matrix**2 / self.dependence_distance**2)
            distance_matrix.diagonal(dim1=-1, dim2=-2).fill_(1.0)
                        
            correlation_matrix = distance_matrix.unsqueeze(0).expand(model_samples.shape[0], -1, -1)
                  
            tt_tril = decompose_covariance_matrix(
                tt_std, correlation_matrix)
            tt_tril = torch.tril(tt_tril)
            # tt_tril += torch.diag_embed(torch.full_like(tt, self.tt_obs_std))
            diag_indices = torch.arange(tt_tril.shape[-1], device=tt_tril.device)
            tt_tril[:, diag_indices, diag_indices] = torch.sqrt(
                tt_tril[:, diag_indices, diag_indices]**2 + self.tt_obs_std**2)

            return tt_tril
        else:
            return tt_std**2 + self.tt_obs_std**2


class DataLikelihood(BaseDataLikelihood):
    def __call__(
        self, model_samples, design,
        remove_mean=True
        ):        
        tt = self.forward_function(model_samples.float(), design.float()).double()

        if self.dependence_distance > 0:

            tt_tril = self.cov(tt, model_samples, design)
            
            if remove_mean:
                return WeightedMultivariateNormal(tt.float(), scale_tril=tt_tril.float())
            else:
                return dist.MultivariateNormal(tt.float(), scale_tril=tt_tril.float())
        else:
            tt_std = self.cov(tt, model_samples, design)
            
            if remove_mean:
                return WeightedIndependentNormal(tt.float(), tt_std.sqrt().float())
            else:
                return dist.Independent(dist.Normal(tt.float(), tt_std.sqrt().float()), 1)


class DataLikelihoodAttenuation(BaseDataLikelihood):
    def __init__(self, forward_function, vel_sigma, picking_likelihood, tt_obs_std=0.01, dependence_distance=100.0, DAS=False):
        super().__init__(forward_function, vel_sigma, tt_obs_std, dependence_distance)
        self.p_like = picking_likelihood
        self.DAS = DAS

    def __call__(
        self, nuisance_samples, model_samples,
        design, remove_mean=True):
        
        tt = self.forward_function(model_samples.float(), design.float()).double()
        t = tt.flatten(end_dim=-2)
        m = model_samples.flatten(end_dim=-2)
        n = nuisance_samples.flatten(end_dim=-2)

        if self.dependence_distance > 0:
            tt_tril = self.cov(t, n, m, design)
            
            tt_tril = tt_tril.view(*nuisance_samples.shape[:-1], design.shape[-2], design.shape[-2])

            try:
                tt = t.view(*nuisance_samples.shape[:-1], design.shape[-2])
            except:
                tt = t
            
            if remove_mean:
                return WeightedMultivariateNormal(tt.float(), scale_tril=tt_tril.float())
            else:
                return dist.MultivariateNormal(tt.float(), scale_tril=tt_tril.float())
        else:
            var = self.cov(t, n, m, design)
            var = var.view(*nuisance_samples.shape[:-1], design.shape[-2])
            
            if remove_mean:                
                return WeightedIndependentNormal(tt.float(), var.sqrt().float())
            else:
                return dist.Independent(dist.Normal(tt.float(), var.sqrt().float()), 1)

    def cov(self, tt, nuisance_samples, model_samples, design, theta=None):
        tt_std = torch.sqrt(tt) * self.vel_sigma

        if self.dependence_distance > 0:
            distance_matrix = torch.cdist(design[..., :3], design[..., :3], p=2.0).double()
            distance_matrix = (1 - distance_matrix / self.dependence_distance).clamp(0.0, 0.7)
            distance_matrix.diagonal(dim1=-1, dim2=-2).fill_(1.0)
            
            correlation_matrix = distance_matrix.unsqueeze(0).expand(model_samples.shape[0], -1, -1)
                        
            tt_tril = decompose_covariance_matrix(tt_std, correlation_matrix)
            tt_tril += torch.diag_embed(torch.full_like(tt, self.tt_obs_std))

            if self.DAS:
                theta = self.forward_function(model_samples.float(), design[..., 3:].float())
                p_like = self.p_like(tt, theta)
            else:
                p_like = self.p_like(tt)
            
            mask = (p_like.unsqueeze(-2).expand(-1, -1, tt_tril.shape[-1])
                    >= nuisance_samples.unsqueeze(-1).expand(-1, -1, p_like.shape[-1]))

            #TODO: set correlations to zero where p_like < nuisance_samples
            diag_mask = torch.eye(mask.size(-1), device=mask.device, dtype=torch.bool).unsqueeze(0).expand(mask.size(0), -1, -1)
            tt_tril = tt_tril.masked_fill(diag_mask & (~mask), 0.5)

            return tt_tril
        else:
            tt_var = tt_std**2 + self.tt_obs_std**2
            
            if self.DAS:
                theta = self.forward_function(model_samples.float(), design.float())
                p_like = self.p_like(tt, theta)
            else:
                p_like = self.p_like(tt)

            mask = p_like >= nuisance_samples
            tt_var = tt_var.where(mask, 0.5**2)
            
            return tt_var


#################################################################################
############################## picking likelihood ###############################
#################################################################################

def logistic_picking_likelihood_offset(
    offset,
    a=torch.tensor( 1.03985654    ),
    b=torch.tensor(-6.18137285e-03),
    c=torch.tensor( 5.42879370e+02)):
    
    return a / (1 + torch.exp(-b * (offset - c)))

class logistic_picking_likelihood_tt:

    def __init__(self,
       a=torch.tensor( 1.0 ),
       b=torch.tensor(-30 ),
       c=torch.tensor( 0.35)):
        
        self.a = a
        self.b = b
        self.c = c
    
    
    def __call__(self, tt):
        return self.a / (1 + torch.exp(-self.b * (tt - self.c)))
    
class logistic_picking_likelihood_tt_theta:

    def __init__(self,
       a=torch.tensor( 1.0 ),
       b=torch.tensor(-30 ),
       c=torch.tensor( 0.35)):
        
        self.a = a
        self.b = b
        self.c = c
    
    
    def __call__(self, tt, theta):
        tt_dependence = self.a / (1 + torch.exp(-self.b * (tt - self.c)))
        angle_dependence = torch.cos(theta)**2

        angle_dependence = angle_dependence.where(
            angle_dependence.isfinite(), torch.ones_like(angle_dependence))
        return tt_dependence * angle_dependence