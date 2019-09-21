import torch
import torch.nn as nn

def make_embedding_layer(in_features, sz_embedding, weight_init = None):
    embedding_layer = torch.nn.Linear(in_features, sz_embedding)
    if weight_init != None: 
        weight_init(embedding_layer.weight)
    return embedding_layer

def bn_inception_weight_init(weight):
    import scipy.stats as stats
    stddev = 0.001
    X = stats.truncnorm(-2, 2, scale=stddev)
    values = torch.Tensor(
        X.rvs(weight.data.numel())
    ).resize_(weight.size())
    weight.data.copy_(values)

def embed(model, sz_embedding, normalize_output = True, net_id = 0):
    if net_id == "bn_inception_v2":
        in_features = model.last_linear.in_features
    if net_id == "densenet201":
        in_features = model.classifier.in_features
    model.embedding_layer = make_embedding_layer(
        in_features,
        sz_embedding,
        weight_init = bn_inception_weight_init
    )
    global_pool = nn.AvgPool2d (
            7, stride=1, padding=0, ceil_mode=True, count_include_pad=True
        )
    def forward(x):
        # split up original logits and forward methods
        x = model.features(x)
        x = global_pool(x)
        x = x.view(x.size(0), -1)
        x = model.embedding_layer(x)
        #for i in model.embedding_layer.named_parameters():
        #    print(i)
        if normalize_output == True:
            x = torch.nn.functional.normalize(x, p = 2, dim = 1)
        return x
    model.forward = forward
