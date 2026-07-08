"""Height-based colormap for point cloud visualization."""
import numpy as np

GRAY_OUT_COLOR = np.array([0.65, 0.65, 0.65])


def height_colormap_colors(points):
    """Map each point to an RGB color based on its Z height, using a
    blue (low) -> red (high) hue sweep normalized over the full Z range of
    `points`.
    """
    z = points[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())
    if z_max > z_min:
        t = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    else:
        t = np.zeros_like(z)

    hue = (1.0 - t) * (2.0 / 3.0)  # 0.66 (blue) -> 0.0 (red)
    saturation = np.ones_like(hue)
    value = np.ones_like(hue)

    h6 = hue * 6.0
    i = np.floor(h6).astype(np.int64) % 6
    f = h6 - np.floor(h6)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * f)
    t_ = value * (1.0 - saturation * (1.0 - f))

    rgb = np.zeros((z.shape[0], 3))
    choices = [
        np.stack([value, t_, p], axis=1),
        np.stack([q, value, p], axis=1),
        np.stack([p, value, t_], axis=1),
        np.stack([p, q, value], axis=1),
        np.stack([t_, p, value], axis=1),
        np.stack([value, p, q], axis=1),
    ]
    for k in range(6):
        mask = i == k
        rgb[mask] = choices[k][mask]
    return rgb
