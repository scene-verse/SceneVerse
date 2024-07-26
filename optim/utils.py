def no_decay_param_group(parameters, lr):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    decay_params = []
    no_decay_params = []
    for n, p in parameters:
        if p.requires_grad == False:
            continue
        if not any(nd in n for nd in no_decay):
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    optimizer_grouped_parameters = [
        {'params': decay_params,
         'weight_decay': 0.01, 'lr': lr},
        {'params': no_decay_params,
         'weight_decay': 0.0, 'lr': lr}
    ]
    return optimizer_grouped_parameters
