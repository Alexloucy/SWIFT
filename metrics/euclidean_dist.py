import torch


def euclidean_dist(x, y):
    """
    Compute euclidean distance between a set of vector pairs.  
    
    Args:
        x (torch.Tensor): A tensor with shape (m, d).
        y (torch.Tensor): A tensor with shape (n, d).

    Returns:
        dist (torch.Tensor): A distance matrix with shape (m, n).
    """
    m, n = x.size(0), y.size(0)
    xx_sum = torch.pow(x, 2).sum(1, keepdim = True).expand(m, n)        # Shape: (m, n)
    yy_sum = torch.pow(y, 2).sum(1, keepdim = True).expand(n, m).t()    # Shape: (m, n)
    dist = xx_sum + yy_sum
    dist = dist - 2 * torch.matmul(x, y.t())    # Shape: (m, n)
    dist = dist.clamp(min = 1e-12).sqrt()    # for numerical stability
    return dist