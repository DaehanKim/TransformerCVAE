# TransformerCVAE

This repository contains source code for paper [Transformer-based Conditional Variational Autoencoder for Controllable Story Generation](https://arxiv.org/abs/2101.00828):

```
@article{fang2021transformer,
  title={Transformer-based Conditional Variational Autoencoder for Controllable Story Generation},
  author={Fang, Le and Zeng, Tao and Liu, Chaochun and Bo, Liefeng and Dong, Wen and Chen, Changyou},
  journal={arXiv preprint arXiv:2101.00828},
  year={2021}
}
```


### 전체 튜닝할 때 amp out of index 에러

```
# apex/amp/utils.py 아래와 같이 comment out 하기

def cached_cast(cast_fn, x, cache):
    if is_nested(x):
        return type(x)([cached_cast(y) for y in x])
    if x in cache:
        cached_x = cache[x]
        if x.requires_grad and cached_x.requires_grad:
            pass
            # Make sure x is actually cached_x's autograd parent.
            # if cached_x.grad_fn.next_functions[1][0].variable is not x:
            #     raise RuntimeError("x and cache[x] both require grad, but x is not "
            #                        "cache[x]'s parent.  This is likely an error.")
        # During eval, it's possible to end up caching casted weights with
        # requires_grad=False.  On the next training iter, if cached_x is found
        # and reused from the cache, it will not actually have x as its parent.
        # Therefore, we choose to invalidate the cache (and force refreshing the cast)
        # if x.requires_grad and cached_x.requires_grad do not match.
        #
        # During eval (i.e. running under with torch.no_grad()) the invalidation
        # check would cause the cached value to be dropped every time, because
        # cached_x would always be created with requires_grad=False, while x would
        # still have requires_grad=True.  This would render the cache effectively
        # useless during eval.  Therefore, if we are running under the no_grad()
        # context manager (torch.is_grad_enabled=False) we elide the invalidation
        # check, and use the cached value even though its requires_grad flag doesn't
        # match.  During eval, we don't care that there's no autograd-graph
        # connection between x and cached_x.
        if torch.is_grad_enabled() and x.requires_grad != cached_x.requires_grad:
            del cache[x]
        else:
            return cached_x

    casted_x = cast_fn(x)
    cache[x] = casted_x
    return casted_x
```
