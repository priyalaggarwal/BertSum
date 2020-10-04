"""
 code has been taken from Annotated Transformers :
https://nlp.seas.harvard.edu/2018/04/03/attention.html

TODO: Checkpoint functionality
"""

import torch


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, original_lr, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.original_lr = original_lr
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.original_lr * min(step ** (-0.5), step * self.warmup ** (-1.5))


class OptimizerArgs:
    def __init__(self, lr, warmup_steps, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.beta1 = beta1
        self.beta2 = beta2


def _get_params(model_params):
    params = []
    for k, p in model_params:
        if p.requires_grad:
            params.append(p)
    return params

def _get_optim(params, args):
    return NoamOpt(original_lr=args.lr, warmup=args.warmup_steps,
                   optimizer=torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-9))

# Using defaults from the BertSum code
def optim_bert(args, model):
    # print("In optim bert")
    # print(model.named_parameters())
    model_params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert_model')]
    # print(model_params)
    params = _get_params(model_params)
    return _get_optim(params, args)


def optim_decoder(args, model):
    model_params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert_model')]
    params = _get_params(model_params)
    return _get_optim(params, args)


# def get_std_opt(model):
#     return NoamOpt(model.src_embed[0].d_model, 2, 4000,
#                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.999), eps=1e-9))

# def optim(args, model):
#     params = _get_params(list(model.named_parameters()))
#     return NoamOpt(original_lr=args.lr, warmup=args.warmup_steps,
#                    optimizer=torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-9))