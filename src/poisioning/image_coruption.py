import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray


def plasma_fractal(
    map_size: int,
    wibble_decay=3,
    device: str = "cpu",
    *,
    key: PRNGKeyArray | None = None,
) -> Array:
    # device = jax.devices()[0] if device == "cpu" else jax.devices()[1]
    maparray = jnp.empty((map_size, map_size), dtype=jnp.float32)
    maparray = maparray.at[0, 0].set(0)
    stepsize = map_size
    wibble = 1.

    def wibblemean(array, *, key: PRNGKeyArray):
        denom = (
            4
            + wibble
            * jax.random.uniform(
                key=key,
                shape=array.shape,
                minval=-wibble,
                maxval=wibble,
                dtype=jnp.float32,
            )
        )
        denom = jnp.where(jnp.abs(denom) < 1e-6, 1e-6, denom)
        return array / denom

    def fillsquares(maparray: Array, wibble, key):
        cornerref = maparray[0:map_size:stepsize, 0:map_size:stepsize]
        squareaccum = (
            cornerref
            + jnp.roll(cornerref, shift=1, axis=0)
            + jnp.roll(cornerref, shift=1, axis=1)
        )
        maparray = maparray.at[
            stepsize // 2 : map_size : stepsize, stepsize // 2 : map_size : stepsize
        ].set(wibblemean(squareaccum, key=key))
        return maparray

    def filldiamonds(maparray: Array, wibble, key):
        drgrid = maparray[
            stepsize // 2 : map_size : stepsize, stepsize // 2 : map_size : stepsize
        ]
        ulgrid = maparray[0:map_size:stepsize, 0:map_size:stepsize]
        ldrsum = drgrid + jnp.roll(drgrid, shift=1, axis=0)
        lulsum = ulgrid + jnp.roll(ulgrid, shift=-1, axis=1)
        ltsum = ldrsum + lulsum
        maparray = maparray.at[
            0:map_size:stepsize, stepsize // 2 : map_size : stepsize
        ].set(wibblemean(ltsum, key=key))
        tdrsum = drgrid + jnp.roll(drgrid, shift=1, axis=0)
        tulsum = ulgrid + jnp.roll(ulgrid, shift=-1, axis=1)
        tlsum = tdrsum + tulsum
        maparray = maparray.at[
            stepsize // 2 : map_size : stepsize, 0:map_size:stepsize
        ].set(wibblemean(tlsum, key=key))
        return maparray

    def body(i, carry):
        maparray, wibble, key = carry
        stepsize = map_size >> i
        key, subkey1, subkey2 = jax.random.split(key, 3)
        maparray = fillsquares(maparray, wibble, subkey1)
        maparray = filldiamonds(maparray, wibble, subkey2)
        wibble /= wibble_decay
        return (maparray, wibble, key)

    maparray, _, _ = jax.lax.fori_loop(
        1,
        int(jnp.log2(map_size)) + 1,
        body,
        (maparray, wibble, key),
    )
    maparray = maparray - maparray.min()
    max_val = jnp.max(maparray)
    return maparray / jnp.where(max_val < 1e-8, 1., max_val)


# Helper functions
def guassian_kernal(r, alias_blur=0.1, dtype=jnp.float32):
    L = jnp.arange(-r, r + 1, dtype=dtype)
    X, Y = jnp.meshgrid(L, L)
    phiX = jnp.exp(-0.05 * (X**2 + Y**2) / alias_blur**2)
    return X, Y, phiX / phiX.sum()


def disk(r, alias_blur=0.1, dtype=jnp.float32):
    X, Y, weight = guassian_kernal(r, alias_blur, dtype)
    aliased_disk = jnp.array(X**2 + Y**2 <= r**2, dtype=dtype)
    aliased_disk = aliased_disk / aliased_disk.sum()
    return jax.scipy.signal.correlate(aliased_disk, weight, mode="valid", method="fft")


def clipped_zoom(
    image: Array, zoom_factor: float, *, key: PRNGKeyArray | None = None
) -> Array:
    h, w = image.shape[:2]
    ch = int(jnp.ceil(h / zoom_factor))
    cw = int(jnp.ceil(w / zoom_factor))
    top = (h - ch) // 2
    left = (w - cw) // 2
    cropped = image[top : top + ch, left : left + cw]
    return jax.image.resize(cropped, (h, w, image.shape[2]), method="linear")


def rgb2hsv(x: Array) -> Array:
    # x = x / 255.0
    r, g, b = jnp.expand_dims(x[0, ...], axis=0), jnp.expand_dims(x[1, ...], axis=0), jnp.expand_dims(x[2, ...], axis=0)
    max_val = jnp.max(x, axis=0, keepdims=True)
    min_val = jnp.min(x, axis=0, keepdims=True)
    delta = max_val - min_val
    h = jnp.where(
        delta == 0,
        0.0,
        jnp.where(
            max_val == r,
            (g - b) / delta % 6,
            jnp.where(max_val == g, (b - r) / delta + 2, (r - g) / delta + 4),
        ),
    )
    s = jnp.where(max_val == 0, 0.0, delta / max_val) 
    hsv = jnp.concatenate([h, s, max_val], axis=0)
    return jnp.clip(hsv, 0, 1)


def hsv2rgb(x: Array) -> Array:
    h, s, v = jnp.expand_dims(x[0, ...], axis=0), jnp.expand_dims(x[1, ...], axis=0), jnp.expand_dims(x[2, ...], axis=0)
    i = (h * 6).astype(jnp.int32) % 6
    f = h * 6 - 1
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    rgb = jnp.concatenate(
        [
            jnp.where(
                i == 0,
                v,
                jnp.where(
                    i == 1,
                    q,
                    jnp.where(i == 2, p, jnp.where(i == 3, p, jnp.where(i == 4, t, v))),
                ),
            ),
            jnp.where(
                i == 0,
                t,
                jnp.where(
                    i == 1,
                    v,
                    jnp.where(i == 2, v, jnp.where(i == 3, q, jnp.where(i == 4, p, p))),
                ),
            ),
            jnp.where(
                i == 0,
                p,
                jnp.where(
                    i == 1,
                    p,
                    jnp.where(i == 2, t, jnp.where(i == 3, v, jnp.where(i == 4, v, q))),
                ),
            ),
        ],
        axis=0,
    )
    return jnp.clip(rgb, 0, 1)


# image corruption functions
def gaussian_noise(x: Array, severity: int = 1, *, key: PRNGKeyArray) -> Array:
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
    x = x / 255.0
    return (jnp.clip(x + jax.random.normal(key, x.shape, dtype=jnp.float32) * c, 0, 1) * 255).astype(jnp.uint8)


def shot_noise(x: Array, severity: int = 1, *, key: PRNGKeyArray) -> Array:
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
    # x = x * 255.0
    return (
        jnp.clip(x + jax.random.poisson(key, lam=(x * c)) / c, 0, 255)
        ).astype(jnp.uint8)


def impulse_noise(x: Array, severity: int = 1, *, key: PRNGKeyArray):
    c = [0.01, 0.02, 0.03, 0.05, 0.07][severity - 1]
    x = x / 255.0
    subkey1, subkey2 = jax.random.split(key)
    salt = jax.random.uniform(subkey1, x.shape, dtype=jnp.float32) < (c / 2)
    pepper = jax.random.uniform(subkey2, x.shape, dtype=jnp.float32) < (c / 2)
    x = jnp.where(salt, 1., x)
    x = jnp.where(pepper, 0., x)
    return (x * 255).astype(jnp.uint8)


def speckle_noise(x: Array, severity: int = 1, *, key: PRNGKeyArray) -> Array:
    c = [0.06, 0.1, 0.12, 0.16, 0.2][severity - 1]
    x = x / 255.0
    return (
        jnp.clip(x + x * jax.random.normal(key, shape=x.shape, dtype=jnp.float32) * c, 0, 1) * 255
    ).astype(jnp.uint8)


def gaussian_blur(x, severity=1, *, key: PRNGKeyArray | None = None):
    c = [0.4, 0.6, 0.7, 0.8, 1][severity - 1]
    x = x / 255
    r = int(4 * c + 0.5)
    X, Y, weight = guassian_kernal(r, alias_blur=c)

    def blur_channel(channel):
        channel = channel[None, None, :, :]
        k = weight[None, None, :, :]
        channel = jnp.pad(channel, ((0, 0), (0, 0), (r, r), (r, r)), mode="reflect")
        channel = jax.lax.conv(channel, k, (1, 1), "valid")
        return channel[0, 0, :, :]

    channels = jnp.moveaxis(x, -1, 0)
    blurred = jax.vmap(blur_channel)(channels)
    return (jnp.clip(jnp.moveaxis(blurred, 0, -1), 0, 1) * 255).astype(jnp.uint8)


# def defocus_blur(x, severity=1):
#     c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]
#     x = np.array(x) / 255.0
#     radius = int(c[0] * x.shape[0])  # scale by image size
#     kernel = disk(r=radius, alias_blur=c[1])
#     channels = [cv2.filter2D(x[:, :, d], -1, kernel) for d in range(3)]
#     return np.clip(np.stack(channels, axis=-1), 0, 1) * 255


# def motion_blur(x, severity=1):
#     c = [(6, 1), (6, 1.5), (6, 2), (8, 2), (9, 2.5)][severity - 1]
#     output = BytesIO()
#     x.save(output, format="PNG")
#     x = MotionImage(blob=output.getvalue())
#     x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
#     x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
#     if x.ndim == 2:
#         x = np.stack([x] * 3, axis=-1)
#     return np.clip(x[..., [2, 1, 0]], 0, 255)


# def jpeg_compression(x, severity=1):
#     c = [80, 65, 58, 50, 40][severity - 1]
#     output = BytesIO()
#     x.save(output, "JPEG", quality=c)
#     return PILImage.open(output)


# def pixelate(x, severity=1):
#     c = [0.95, 0.9, 0.85, 0.75, 0.65][severity - 1]
#     w, h = x.size
#     x = x.resize((int(w * c), int(h * c)), PILImage.BOX)
#     return x.resize((w, h), PILImage.BOX)


# def frost(x, severity=1):
#     c = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][severity - 1]
#     H, W = np.array(x).shape[:2]
#     idx = np.random.randint(3)
#     filename = [f"./src/poison/frost{i}.jpg" for i in range(4, 7)][idx]
#     frost_img = cv2.imread(filename)
#     if frost_img is None:
#         raise FileNotFoundError(f"Failed to load image at: {filename}")
#     orig_h, orig_w = frost_img.shape[:2]

#     # Compute scale factor to guarantee resized image >= (H, W)
#     scale_factor = max(H / orig_h, W / orig_w, 1.0)
#     frost_img = cv2.resize(frost_img, (0, 0), fx=scale_factor, fy=scale_factor)

#     # Now safe to crop
#     fh, fw = frost_img.shape[:2]
#     y0 = np.random.randint(0, fh - H + 1)
#     x0 = np.random.randint(0, fw - W + 1)
#     frost_crop = frost_img[y0 : y0 + H, x0 : x0 + W][..., ::-1] / 255.0
#     x_arr = np.array(x) / 255.0
#     return np.clip(c[0] * x_arr + c[1] * frost_crop, 0, 1) * 255


# def snow(x, severity=1, *, key: PRNGKeyArray):
#     c = [
#         (0.1, 0.2, 1, 0.6, 8, 3, 0.95),
#         (0.1, 0.2, 1, 0.5, 10, 4, 0.9),
#         (0.15, 0.3, 1.75, 0.55, 10, 4, 0.9),
#         (0.25, 0.3, 2.25, 0.6, 12, 6, 0.85),
#         (0.3, 0.3, 1.25, 0.65, 14, 12, 0.8),
#     ][severity - 1]
    # x_arr = x / 255.0--
#     H, W = x_arr.shape[:2]
#     snow_layer = c[0] + jax.random.normal(key, shape=(H, W))[..., None] * c[1]
#     snow_layer = clipped_zoom(snow_layer, c[2])
#     snow_layer[snow_layer < c[3]] = 0
#     img = PILImage.fromarray((snow_layer.squeeze() * 255).astype(np.uint8))
#     buf = BytesIO()
#     img.save(buf, format="PNG")
#     snow_wand = MotionImage(blob=buf.getvalue())
#     snow_wand.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
#     arr = (
#         cv2.imdecode(
#             np.frombuffer(snow_wand.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED
#         )
#         / 255.0
#     )
#     arr = arr[..., None]
#     gray = cv2.cvtColor(x_arr, cv2.COLOR_RGB2GRAY)[..., None]
#     x_mod = c[6] * x_arr + (1 - c[6]) * np.maximum(x_arr, gray * 1.5 + 0.5)
#     return np.clip(x_mod + arr + np.rot90(arr, 2), 0, 1) * 255


def fog(x, severity=1, *, key: PRNGKeyArray):
    c = [(0.2, 3), (0.5, 3), (0.75, 2.5), (1, 2), (1.5, 1.75)][severity - 1]
    x = x / 255.0
    H, W = x.shape[:2]
    max_val = x.max()
    fog_layer = plasma_fractal(map_size=256, wibble_decay=int(c[1]), key=key)
    fog_crop = fog_layer[:H, :W][..., None]
    x_mod = x + c[0] * fog_crop
    return (jnp.clip(x_mod * max_val / (max_val + c[0]), 0, 1) * 255).astype(jnp.uint8)


# def spatter(x, severity=1):
#     c = [
#         (0.62, 0.1, 0.7, 0.7, 0.5, 0),
#         (0.65, 0.1, 0.8, 0.7, 0.5, 0),
#         (0.65, 0.3, 1, 0.69, 0.5, 0),
#         (0.65, 0.1, 0.7, 0.69, 0.6, 1),
#         (0.65, 0.1, 0.5, 0.68, 0.6, 1),
#     ][severity - 1]

#     x = np.array(x, dtype=np.float32) / 255.0
#     H, W = x.shape[:2]
#     liquid = np.random.normal(loc=c[0], scale=c[1], size=(H, W))

#     liquid = gaussian(liquid, sigma=c[2])
#     liquid[liquid < c[3]] = 0

#     if c[5] == 0:
#         liquid_img = (liquid * 255).astype(np.uint8)
#         dist = 255 - cv2.Canny(liquid_img, 50, 150)
#         dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
#         dist = np.clip(dist, 0, 20)
#         dist = cv2.blur(dist.astype(np.uint8), (3, 3))
#         dist = cv2.equalizeHist(dist)

#         ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
#         dist = cv2.filter2D(dist, cv2.CV_8U, ker)
#         dist = cv2.blur(dist, (3, 3)).astype(np.float32)

#         m = cv2.cvtColor(liquid * dist, cv2.COLOR_GRAY2BGR)
#         m = np.clip(m / m.max() * c[4], 0, 1)

#         color = np.stack(
#             [
#                 np.full_like(m[..., 0], 175 / 255.0),
#                 np.full_like(m[..., 1], 238 / 255.0),
#                 np.full_like(m[..., 2], 238 / 255.0),
#             ],
#             axis=-1,
#         )

#         return np.clip(x + m * color, 0, 1) * 255

#     else:
#         m = (liquid > c[3]).astype(np.float32)
#         m = gaussian(m, sigma=c[4])
#         m[m < 0.8] = 0

#         mud = np.stack(
#             [
#                 np.full_like(x[..., 0], 63 / 255.0),
#                 np.full_like(x[..., 1], 42 / 255.0),
#                 np.full_like(x[..., 2], 20 / 255.0),
#             ],
#             axis=-1,
#         )

#         color = mud * m[..., None]
#         x *= 1 - m[..., None]
#         return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1, *, key: PRNGKeyArray | None = None):
    c = [0.75, 0.5, 0.4, 0.3, 0.15][severity - 1]
    x = x / 255.0
    means = jnp.mean(x, axis=(0, 1), keepdims=True)
    return (jnp.clip((x - means) * c + means, 0, 1) * 255).astype(jnp.uint8)


def brightness(x, severity=1, *, key: PRNGKeyArray | None = None):
    c = [0.05, 0.1, 0.15, 0.2, 0.3][severity - 1]
    x = x / 255.0
    hsv = rgb2hsv(x)
    hsv = hsv.at[..., 2].set(jnp.clip(hsv[..., 2] + c, 0, 1))
    return (hsv2rgb(hsv) * 255).astype(jnp.uint8)


def saturate(x, severity=1, *, key: PRNGKeyArray | None = None):
    c = [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)][severity - 1]
    x = x / 255.0
    hsv = rgb2hsv(x)
    hsv = hsv.at[..., 1].set(jnp.clip(hsv[..., 1] * c[0] + c[1], 0, 1))
    return (hsv2rgb(hsv) * 255).astype(jnp.uint8)


# def elastic_transform(image, severity=1):
#     IMSIZE = image.size[0]  # handles 32 or 64 automatically
#     c = [
#         (IMSIZE * 0, IMSIZE * 0, IMSIZE * 0.08),
#         (IMSIZE * 0.05, IMSIZE * 0.2, IMSIZE * 0.07),
#         (IMSIZE * 0.08, IMSIZE * 0.06, IMSIZE * 0.06),
#         (IMSIZE * 0.1, IMSIZE * 0.04, IMSIZE * 0.05),
#         (IMSIZE * 0.1, IMSIZE * 0.03, IMSIZE * 0.03),
#     ][severity - 1]

#     image = np.array(image, dtype=np.float32) / 255.0
#     shape = image.shape
#     shape_size = shape[:2]

#     center_square = np.float32(shape_size) // 2
#     square_size = min(shape_size) // 3
#     pts1 = np.float32(
#         [
#             center_square + square_size,
#             [center_square[0] + square_size, center_square[1] - square_size],
#             center_square - square_size,
#         ]
#     )
#     pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)

#     M = cv2.getAffineTransform(pts1, pts2)
#     image = cv2.warpAffine(
#         image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101
#     )

#     dx = (
#         gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect") * c[0]
#     ).astype(np.float32)
#     dy = (
#         gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect") * c[0]
#     ).astype(np.float32)
#     dx, dy = dx[..., None], dy[..., None]

#     x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
#     indices = (
#         np.reshape(y + dy, (-1, 1)),
#         np.reshape(x + dx, (-1, 1)),
#         np.reshape(z, (-1, 1)),
#     )

#     return (
#         np.clip(
#             map_coordinates(image, indices, order=1, mode="reflect").reshape(shape),
#             0,
#             1,
#         )
#         * 255
#     )


# def glass_blur(x, severity=1):
#     # (sigma, max_delta, iterations)
#     c = [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2)][
#         severity - 1
#     ]
#     sigma, delta, iterations = c

#     x = np.array(x) / 255.0
#     x = np.clip(gaussian(x, sigma=sigma, channel_axis=-1), 0, 1)
#     x = np.uint8(x * 255)

#     H, W = x.shape[:2]

#     for _ in range(iterations):
#         for h in range(delta, H - delta):
#             for w in range(delta, W - delta):
#                 dx, dy = np.random.randint(-delta, delta + 1, size=2)
#                 h_new, w_new = h + dy, w + dx
#                 if 0 <= h_new < H and 0 <= w_new < W:
#                     tmp = x[h, w].copy()
#                     x[h, w] = x[h_new, w_new]
#                     x[h_new, w_new] = tmp

    # x = x / 255.0
#     return np.clip(gaussian(x, sigma=sigma, channel_axis=-1), 0, 1) * 255


def zoom_blur(x, severity=1, *, key: PRNGKeyArray | None = None):
    c = [
        np.arange(1, 1.06, 0.01),
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.01),
        np.arange(1, 1.26, 0.01),
    ][severity - 1]
    x = x / 255.0
    out = jnp.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return (jnp.clip(x, 0, 1) * 255).astype(jnp.uint8)


def clean(x, severity=1, *, key: PRNGKeyArray | None = None):
    return x


corruption_dict = {
    'gaussian_noise': gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    # 'defocus_blur': defocus_blur,
    "gaussian_blur": gaussian_blur,
    # 'motion_blur': motion_blur,
    "speckle_noise": speckle_noise,
    # 'jpeg_compression': jpeg_compression,
    # 'pixelate': pixelate,
    # 'frost': frost,
    # 'snow': snow,
    "fog": fog,
    # 'spatter': spatter,
    "contrast": contrast,
    "brightness": brightness,
    "saturate": saturate,
    # 'elastic_transform': elastic_transform,
    # 'glass_blur': glass_blur,
    "zoom_blur": zoom_blur,
    "clean": clean,
}


def corrupt_image(image: Array, *, key: PRNGKeyArray | None = None) -> Array:
    return image
