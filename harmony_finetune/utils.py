def normalize_vector(vec):
    vec = vec / (vec.pow(2).sum(-1, keepdim=True).pow(0.5) + 1e-9)
    return vec

def backprop_normalize(x):
    original_shape = x.shape
    x = x.flatten(1)
    x = x / (x.abs().sum(-1, keepdim=True) + 1e-9)
    x = x.view(*original_shape)
    
    return x

def backprop_normalize_to_one(x):
    original_shape = x.shape
    x = x.flatten(1)
    x = x / (x.abs().max(-1, keepdim=True)[0] + 1e-9)
    x = x.view(*original_shape)
    
    return x

def backprop_normalize_to_one_with_detach(x):
    original_shape = x.shape
    x = x.flatten(1)
    x = x / (x.abs().max(-1, keepdim=True)[0].detach() + 1e-9)
    x = x.view(*original_shape)
    
    return x