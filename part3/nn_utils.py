"""
Neural network utilities for Transformer implementation.
Contains basic building blocks: softmax, cross-entropy, gradient clipping, token accuracy, perplexity.
"""
import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute softmax along the specified dimension.
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute softmax (default: -1)
    
    Returns:
        Tensor of same shape as input with softmax applied along dim
    """
    # TODO: Implement numerically stable softmax. You can re-use the same one 
    # used in part 2. But for this problem, you need to implement a numerically stable version to pass harder tests.
    x_safe = x - torch.max(x, dim=dim, keepdim=True).values
    result = torch.exp(x_safe) / torch.sum(torch.exp(x_safe), dim=dim, keepdim=True)
    result = result.nan_to_num(0.0)
    return result


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Unnormalized log probabilities of shape (N, C) where N is batch size
                and C is number of classes
        targets: Ground truth class indices of shape (N,)
    
    Returns:
        Scalar tensor containing the mean cross-entropy loss
    """
    # TODO: Implement cross-entropy loss
    batch_size = logits.shape[0]
    #log_prob = torch.log(softmax(logits, dim=-1)) # (N,C)

    ## log(softmax(x)) = log(exp(x_i) / sum(exp(x)))
    #                  = x_i - log(sum(exp(x)))
    # 直接用减法，避免了中间 softmax 产生极小值再取 log 的问题
    ## 你原来的写法：
    #log(softmax([1000, 0, 0]))  # softmax → [1.0, 0.0, 0.0] → log → [0, -inf, -inf] 💥

    # 直接算：
    # log_softmax(x_0) = 1000 - log(exp(1000) + exp(0) + exp(0)) = 1000 - 1000 ≈ 0 ✅
    x_safe = logits - torch.max(logits, dim=-1, keepdim=True).values #torch.exp(x) 在 x 很大时会溢出变成 inf：
    log_prob = x_safe - torch.log(torch.exp(x_safe).sum(dim=-1, keepdim=True))
    correct_log_prob = log_prob[torch.arange(batch_size), targets] #(N,) pick up the correct one based on targets
    
    return -correct_log_prob.mean()
    


def gradient_clipping(parameters, max_norm: float) -> Tensor:
    """
    Clip gradients of parameters by global norm.
    
    Args:
        parameters: Iterable of parameters with gradients
        max_norm: Maximum allowed gradient norm
    
    Returns:
        The total norm of the gradients before clipping
    """
    # TODO: Implement gradient clipping
    
    parameters = [p for p in parameters if p.grad is not None]
    norm = torch.sqrt(sum(p.grad.norm()**2 for p in parameters))

    if norm > max_norm:
        scale = max_norm/norm
        for p in parameters:
            p.grad *= scale 
    return norm


def token_accuracy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute token-level accuracy for language modeling.
    
    Computes the fraction of tokens where the predicted token (argmax of logits)
    matches the target token, ignoring positions where target equals ignore_index.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing accuracy (default: -100)
    
    Returns:
        Scalar tensor containing the accuracy (between 0 and 1)
    
    Example:
        >>> logits = torch.tensor([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2], [1.0, 0.5, 2.5]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> token_accuracy(logits, targets)
        tensor(1.)  # All predictions correct: argmax gives [0, 1, 2]
        
        >>> logits = torch.tensor([[2.0, 1.0], [0.1, 3.0], [1.0, 0.5]])
        >>> targets = torch.tensor([1, 1, 0])
        >>> token_accuracy(logits, targets)
        tensor(0.6667)  # 2 out of 3 correct
    """
    # TODO: Implement token accuracy
    
    pred = logits.argmax(dim=-1)
    mask = targets != ignore_index

    return sum((pred == targets) & mask) / sum(mask)



def perplexity(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute perplexity for language modeling.
    
    Perplexity is defined as exp(cross_entropy_loss). It measures how well the
    probability distribution predicted by the model matches the actual distribution
    of the tokens. Lower perplexity indicates better prediction.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing perplexity (default: -100)
    
    Returns:
        Scalar tensor containing the perplexity (always >= 1)
    
    Example:
        >>> # Perfect predictions (one-hot logits matching targets)
        >>> logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(1.0001)  # Close to 1 (perfect)
        
        >>> # Uniform predictions (high uncertainty)
        >>> logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(3.)  # Equal to vocab_size (worst case for uniform)
    """
    # TODO: Implement perplexity
    mask = targets != ignore_index
    targets = targets[mask]
    logits = logits[mask, :]
    return torch.exp(cross_entropy(logits, targets))
    
