import torch.optim as optim

from Laplacian_Pyramid import *
from Style_Transfer import *


# Optimization Function
# This function optimizes the style transfer by adjusting the image to minimize the content and style losses at various scales.
def optimize(result, content, style, scale, content_weight, lr, extractor):
    # torch.autograd.set_detect_anomaly(True)
    result_pyramid = make_laplace_pyramid(result, 5)
    result_pyramid = [l.data.requires_grad_() for l in result_pyramid]

    opt_iter = 200
    # if scale == 1:
    #     opt_iter = 800

    # use rmsprop
    optimizer = optim.RMSprop(result_pyramid, lr=lr)

    # extract features for content
    feat_content = extractor(content) # 

    stylized = fold_laplace_pyramid(result_pyramid)
    # let's ignore the regions for now
    # some inner loop that extracts samples
    feat_style = None
    for i in range(5):
        with torch.no_grad():
            # r is region of interest (mask)
            feat_e = extractor.forward_samples_hypercolumn(style, samps=1000)
            feat_style = feat_e if feat_style is None else torch.cat((feat_style, feat_e), dim=2)
    # feat_style.requires_grad_(False)

    # init indices to optimize over
    xx, xy = sample_indices(feat_content[0], feat_style) # 0 to sample over first layer extracted
    for it in range(opt_iter):
        optimizer.zero_grad()

        stylized = fold_laplace_pyramid(result_pyramid)
        # original code has resample here, seems pointless with uniform shuffle
        # ...
        # also shuffle them every y iter
        if it % 1 == 0 and it != 0:
            np.random.shuffle(xx)
            np.random.shuffle(xy)
        feat_result = extractor(stylized)

        loss = calculate_loss(feat_result, feat_content, feat_style, [xx, xy], content_weight)
        loss.backward()
        optimizer.step()
    return stylized