def compute_accuracy(output, target, topk = (1, )):
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (torch.Tensor): A prediction matrix with a shape (batch_size, num_classes).
        target (torch.LongTensor): The ground truth labels with a shape (batch_size).
        topk (tuple, optional): The accuracy at top-k will be computed. For example, 
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: The accuracy at top-k.

    Example:
        >>> output = torch.tensor([[0.1, 0.2, 0.7, 0.4], [0.8, 0.1, 0.1, 0.5], [0.2, 0.9, 0.5, 0.1]])
        >>> target = torch.tensor([2, 3, 1])
        >>> compute_accuracy(output, target, topk = (1, 2))
            [tensor([66.6667]), tensor([100.])]
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)    # pred (indices): tensor of shape (batch_size, maxk)
    pred = pred.t()    # transpose, shape: (maxk, batch_size)
    correct = pred.eq(target.view(1, -1).expand_as(pred))    # shape: (maxk, batch_size), bool tensor

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


