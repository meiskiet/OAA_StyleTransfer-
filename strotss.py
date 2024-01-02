import torch

from time import time
from argparse import ArgumentParser

from Vgg16_Extractor import Vgg16_Extractor

from Image_Process import *
from Laplacian_Pyramid import *
from Style_Transfer import *
from Optimization import *

def strotss(content_pil, style_pil, content_weight=1.0*16.0, device='cuda:0', space='uniform'):
    content_np = pil_to_np(content_pil)
    style_np = pil_to_np(style_pil)
    content_full = np_to_tensor(content_np, space).to(device)
    style_full = np_to_tensor(style_np, space).to(device)

    lr = 2e-3
    extractor = Vgg16_Extractor(space=space).to(device)

    scale_last = max(content_full.shape[2], content_full.shape[3])
    scales = []
    for scale in range(10):
        divisor = 2**scale
        if min(content_pil.width, content_pil.height) // divisor >= 33:
            scales.insert(0, divisor)
    
    for scale in scales:
        # rescale content to current scale
        content = tensor_resample(content_full, [ content_full.shape[2] // scale, content_full.shape[3] // scale ])
        style = tensor_resample(style_full, [ style_full.shape[2] // scale, style_full.shape[3] // scale ])
        print(f'Optimizing at resoluton [{content.shape[2]}, {content.shape[3]}]')

        # upsample or initialize the result
        if scale == scales[0]:
            # first
            result = laplacian(content) + style.mean(2,keepdim=True).mean(3,keepdim=True)
        elif scale == scales[-1]:
            # last 
            result = tensor_resample(result, [content.shape[2], content.shape[3]])
            lr = 1e-3
        else:
            result = tensor_resample(result, [content.shape[2], content.shape[3]]) + laplacian(content)

        # do the optimization on this scale
        result = optimize(result, content, style, scale, content_weight=content_weight, lr=lr, extractor=extractor)

        # next scale lower weight
        content_weight /= 2.0

    clow = -1.0 if space == 'uniform' else -1.7
    chigh = 1.0 if space == 'uniform' else 1.7
    result_image = tensor_to_np(tensor_resample(torch.clamp(result, clow, chigh), [content_full.shape[2], content_full.shape[3]])) # 
    # renormalize image
    result_image -= result_image.min()
    result_image /= result_image.max()
    return np_to_pil(result_image * 255.)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("content", type=str)
    parser.add_argument("style", type=str)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="strotss.png")
    parser.add_argument("--device", type=str, default="cuda:0")
    # uniform ospace = optimization done in [-1, 1], else imagenet normalized space
    parser.add_argument("--ospace", type=str, default="uniform", choices=["uniform", "vgg"])
    parser.add_argument("--resize_to", type=int, default=512)
    args = parser.parse_args()

    # make 256 the smallest possible long side, will still fail if short side is <
    if args.resize_to < 2**8:
        print("Resulution too low.")
        exit(1)

    content_pil, style_pil = pil_loader(args.content), pil_loader(args.style)
    content_weight = args.weight * 16.0

    device = args.device

    start = time()
    result = strotss(pil_resize_long_edge_to(content_pil, args.resize_to), 
                     pil_resize_long_edge_to(style_pil, args.resize_to), content_weight, device, args.ospace)
    result.save(args.output)
    print(f'Done in {time()-start:.3f}s')


print(torch.cuda.is_available())