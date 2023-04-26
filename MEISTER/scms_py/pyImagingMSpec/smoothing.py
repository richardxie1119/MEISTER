__author__ = 'palmer'
# every method in smoothing should accept (im,**args)
def median(im, **kwargs):
    from scipy import ndimage
    im = ndimage.filters.median_filter(im,**kwargs)
    return im

def hot_spot_removal(im, q=99.):
    import numpy as np
    xic_q = np.percentile(im, q)
    im[im > xic_q] = xic_q
    return im

def log(im):
    import numpy as np
    return np.log(im)