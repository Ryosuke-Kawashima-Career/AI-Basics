# Layer Design
class BatchNorm(nn.Module):
    def __init__(self, shape, epsilon=np.float32(1e-5)):
        """y = gemma * x + beta
        """
        self.gemma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.epsilon = epsilon
        super().__init__()
    
    def forward(self, x):
        """Calculates the normalized input batch-wise
        """
        # (N, C, H, W) -> 0: batch 2: Height 3: Weight
        # Except for the channels
        # Step 1: Compute batch mean
            # Reduce over batch (N), height (H), width (W)
            # Keep channel (C) separate
            # Input: (N, C, H, W)
            # Output: (C,)
        x_mean = torch.mean(x, (0, 2, 3), keepdim=True)
        x_std = torch.std(x, (0, 2, 3), keepdim=True)
        # Step 3: Normalize
            # (N, C, H, W) - (C,) = (N, C, H, W) [broadcasting over N, H, W]
            # (N, C, H, W) / (C,) = (N, C, H, W) [broadcasting over N, H, W]
        x_normalized = (x - x_mean) / ((x_std * x_std + self.epsilon) ** 0.5)
        return x_normalized

class Dropout(nn.Module):
    """inhibits overfittings by deactivating some neurons
    Note that it needs to reduce the amount of output by (1 - dropout)
    """
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        super().__init__()
    
    def forward(self, x):
        """Calculates masking effects
        """
        if self.training:
            self.mask = torch.rand(*x.size()) > dropout_ratio
            return x * self.mask.to(x.device)
        else:
            x * (1 - self.dropout_ratio)
class Conv(nn.Module):
    """Calculates convolution
    """
    def __init__(self, filter_shape, function: lambda x: x, stride=(1, 1), padding=0):
        # filter_shape = (out_channel, in_channel, Height, Width)
        super().__init__()
        fan_in = filter_shape[1] * filter_shape[2] * filter_shape[3]
        fan_out = filter_shape[0] * filter_shape[2] * filter_shape[3]
        self.W = nn.Parameter(torch.tensor(rng.normal(
            0, np.sqrt(6 / (fan_in + fan_out)), size=fileter_shape), dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(filter_shape[0], dtype=torch.float32))
        self.function = function
        self.stride = stride
        self.padding = padding
    
    def forward(self, x):
        """Calculates the convolution
        """
        u = F.conv2d(x, self.W, self.b, self.stride, self.padding)
        return self.function(u)