import torch

class Homogeneous_Forward_Function:
    def __init__(self, v):
        self.v = v
    def __call__(self, m, design):
        
        if m.ndim == 1:
            m_in = m[None, :]
        else:
            m_in = m
            
        if design.ndim == 1:
            design_in = design[None, :]
        else:
            design_in = design
        
        distance = torch.cdist(
            m_in[..., :3].float(),
            design_in[..., :3].float())
                
        # if m.ndim == 1:
        #     distance = distance.squeeze(0)
        
        return distance / self.v

    def theta(self, m, design):
        
        # if m.ndim == 1:
        #     m_in = m[None, :]
        # else:
        #     m_in = m
            
        # if design.ndim == 1:
        #     design_in = design[None, :]
        # else:
        #     design_in = design
        
        dir_vec = design[..., :3] - m[..., :3].unsqueeze(-2)        
        dir_vec = dir_vec / torch.norm(dir_vec, dim=-1, keepdim=True)
        # compute angle between the 3d vectors dir_vec and design_in[..., 3:]
        theta = torch.acos(torch.sum(dir_vec * design[..., 3:], dim=-1))
        
        return theta
    

class TTLookup:
    def __init__(self, models, designs, data, theta=None):
        self.model_space = models.clone()
        self.design_space = designs.clone()
        self.data = data.clone()
        self.theta = theta.clone() if theta is not None else None

    def __call__(self, models, designs):
        
        m = models.flatten(end_dim=-2)
        d = designs.flatten(end_dim=-2)

        out = self.data[m[..., -1].int()][:, d[..., -1].int()]
        out = out.view(*models.shape[:-1], *designs.shape[:-1])

        return out
    
    def theta(self, models, designs):
        
        if self.theta is None:
            raise ValueError('No theta values provided.')
        
        d = designs.flatten(end_dim=-2)
        m = models.flatten(end_dim=-2)
        out = self.theta[m[..., -1].int()][:, d[..., -1].int()]
        out = out.view(*models.shape[:-1], *designs.shape[:-1])

        out = torch.acos(torch.sum(out * designs[..., 3:-1], dim=-1))


        return out
