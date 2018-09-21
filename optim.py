"""
Optimisers.
"""
import math
import torch


class SharedAdam(torch.optim.Adam):
    """
    Adam, but with a shared state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = param.data.new().resize_as_(
                    param.data).zero_()
                state['exp_avg_sq'] = param.data.new().resize_as_(
                    param.data).zero_()

    def share_memory(self):
        """
        Prepare the state for sharing.
        """
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please "
                        "consider SparseAdam instead")
                amsgrad = group['amsgrad']

                state = self.state[param]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], param.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * \
                    math.sqrt(bias_correction2) / bias_correction1

                param.data.addcdiv_(-step_size.item(), exp_avg, denom)

        return loss
