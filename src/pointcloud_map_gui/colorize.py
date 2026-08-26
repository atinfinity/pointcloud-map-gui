"""Height-based colormap for point cloud visualization."""
import numpy as np

GRAY_OUT_COLOR = np.array([0.65, 0.65, 0.65])


def _srgb_encode(linear):
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.abs(linear) ** (1 / 2.4) - 0.055)


def _srgb_decode(encoded):
    return np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)


def fade_to_background(color, alpha, background):
    """Return the opaque color that looks like `color` drawn at `alpha` over
    `background`.

    De-emphasized points used to be a separate geometry with a translucent
    material, which forced a full geometry rebuild whenever a point moved
    between the emphasized and de-emphasized sets. Baking the fade into the
    color instead lets every point live in one geometry whose colors are the
    only thing that changes.

    The blend has to happen in display space, not in the linear values the
    shader takes, or the result comes out far too bright. Filament sRGB-encodes
    point colors on the way out but hands the scene background through roughly
    as given (measured: a 0.35 background renders as 81/255), so only `color`
    gets encoded here.

    One difference remains by construction: real alpha accumulates where
    translucent points overlap, so dense de-emphasized regions used to glow
    brighter than sparse ones. This is a single density-independent color.
    """
    color = np.asarray(color, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    return _srgb_decode(alpha * _srgb_encode(color) + (1.0 - alpha) * background)


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
