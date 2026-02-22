import math
from numbers import Number

import torch
import torch.distributions as dist
from torch import Tensor, Size
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

import zuko

#################################################################################
################################# Empirical ####################################
#################################################################################


class Empirical(Distribution):
    r"""
    Taken from the Pyro library: https://docs.pyro.ai/en/stable/_modules/pyro/distributions/empirical.html#Empirical
     
    Empirical distribution associated with the sampled data. Note that the shape
    requirement for `log_weights` is that its shape must match the leftmost shape
    of `samples`. Samples are aggregated along the ``aggregation_dim``, which is
    the rightmost dim of `log_weights`.

    Example:

    >>> emp_dist = Empirical(torch.randn(2, 3, 10), torch.ones(2, 3))
    >>> emp_dist.batch_shape
    torch.Size([2])
    >>> emp_dist.event_shape
    torch.Size([10])

    >>> single_sample = emp_dist.sample()
    >>> single_sample.shape
    torch.Size([2, 10])
    >>> batch_sample = emp_dist.sample((100,))
    >>> batch_sample.shape
    torch.Size([100, 2, 10])

    >>> emp_dist.log_prob(single_sample).shape
    torch.Size([2])
    >>> # Vectorized samples cannot be scored by log_prob.
    >>> with pyro.validation_enabled():
    ...     emp_dist.log_prob(batch_sample).shape
    Traceback (most recent call last):
    ...
    ValueError: ``value.shape`` must be torch.Size([2, 10])

    :param torch.Tensor samples: samples from the empirical distribution.
    :param torch.Tensor log_weights: log weights (optional) corresponding
        to the samples.
    """

    arg_constraints = {}
    support = dist.constraints.real
    has_enumerate_support = True

    def __init__(self, samples, log_weights, validate_args=None):
        self._samples = samples
        self._log_weights = log_weights
        sample_shape, weight_shape = samples.size(), log_weights.size()
        if (
            weight_shape > sample_shape
            or weight_shape != sample_shape[: len(weight_shape)]
        ):
            raise ValueError(
                "The shape of ``log_weights`` ({}) must match "
                "the leftmost shape of ``samples`` ({})".format(
                    weight_shape, sample_shape
                )
            )
        self._aggregation_dim = log_weights.dim() - 1
        event_shape = sample_shape[len(weight_shape) :]
        self._categorical = dist.Categorical(logits=self._log_weights)
        super().__init__(
            batch_shape=weight_shape[:-1],
            event_shape=event_shape,
            validate_args=validate_args,
        )

    @property
    def sample_size(self):
        """
        Number of samples that constitute the empirical distribution.

        :return int: number of samples collected.
        """
        return self._log_weights.numel()

    def sample(self, sample_shape=torch.Size()):
        sample_idx = self._categorical.sample(
            sample_shape
        )  # sample_shape x batch_shape
        # reorder samples to bring aggregation_dim to the front:
        # batch_shape x num_samples x event_shape -> num_samples x batch_shape x event_shape
        samples = (
            self._samples.unsqueeze(0)
            .transpose(0, self._aggregation_dim + 1)
            .squeeze(self._aggregation_dim + 1)
        )
        # make sample_idx.shape compatible with samples.shape: sample_shape_numel x batch_shape x event_shape
        sample_idx = sample_idx.reshape(
            (-1,) + self.batch_shape + (1,) * len(self.event_shape)
        )
        sample_idx = sample_idx.expand((-1,) + samples.shape[1:])
        return samples.gather(0, sample_idx).reshape(sample_shape + samples.shape[1:])


    def log_prob(self, value):
        """
        Returns the log of the probability mass function evaluated at ``value``.
        Note that this currently only supports scoring values with empty
        ``sample_shape``.

        :param torch.Tensor value: scalar or tensor value to be scored.
        """
        if self._validate_args:
            if value.shape != self.batch_shape + self.event_shape:
                raise ValueError(
                    "``value.shape`` must be {}".format(
                        self.batch_shape + self.event_shape
                    )
                )
        if self.batch_shape:
            value = value.unsqueeze(self._aggregation_dim)
        selection_mask = self._samples.eq(value)
        # Get a mask for all entries in the ``weights`` tensor
        # that correspond to ``value``.
        for _ in range(len(self.event_shape)):
            selection_mask = selection_mask.min(dim=-1)[0]
        selection_mask = selection_mask.type(self._categorical.probs.type())
        return (self._categorical.probs * selection_mask).sum(dim=-1).log()


    def _weighted_mean(self, value, keepdim=False):
        weights = self._log_weights.reshape(
            self._log_weights.size()
            + torch.Size([1] * (value.dim() - self._log_weights.dim()))
        )
        dim = self._aggregation_dim
        max_weight = weights.max(dim=dim, keepdim=True)[0]
        relative_probs = (weights - max_weight).exp()
        return (value * relative_probs).sum(
            dim=dim, keepdim=keepdim
        ) / relative_probs.sum(dim=dim, keepdim=keepdim)

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def mean(self):
        if self._samples.dtype in (torch.int32, torch.int64):
            raise ValueError(
                "Mean for discrete empirical distribution undefined. "
                + "Consider converting samples to ``torch.float32`` "
                + "or ``torch.float64``. If these are samples from a "
                + "`Categorical` distribution, consider converting to a "
                + "`OneHotCategorical` distribution."
            )
        return self._weighted_mean(self._samples)

    @property
    def variance(self):
        if self._samples.dtype in (torch.int32, torch.int64):
            raise ValueError(
                "Variance for discrete empirical distribution undefined. "
                + "Consider converting samples to ``torch.float32`` "
                + "or ``torch.float64``. If these are samples from a "
                + "`Categorical` distribution, consider converting to a "
                + "`OneHotCategorical` distribution."
            )
        mean = self.mean.unsqueeze(self._aggregation_dim)
        deviation_squared = torch.pow(self._samples - mean, 2)
        return self._weighted_mean(deviation_squared)

    @property
    def log_weights(self):
        return self._log_weights

    def enumerate_support(self, expand=True):
        # Empirical does not support batching, so expanding is a no-op.
        return self._samples
    
#################################################################################
############################### TruncatedNormal #################################
#################################################################################    
    
CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Taken from https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py
    
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """
    Taken from https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py
    
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale