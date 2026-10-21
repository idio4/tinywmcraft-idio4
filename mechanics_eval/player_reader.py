"""Player pose from opaque sprite pixels; darkness alone never establishes sleep."""
import numpy as np
from craftax.craftax_classic.constants import TEXTURES

VERSION = 'opaque-player-pose-v1'


def read_player(frame):
    from .reader_v2 import classify
    sprites = np.asarray(TEXTURES[7]['player_textures'], np.float32)
    mask = np.all(sprites[..., 3] == 255, axis=0)
    rgb = sprites[..., :3][:, mask]
    luminance = (rgb * np.array([.299, .587, .114])).sum(-1, keepdims=True)
    base = .5 * (.4 * rgb + .6 * luminance) + .5 * np.array([0, 16, 64])
    slope = rgb - base
    # Include grayscale awake poses as rejection rivals, not sleeping labels.
    sleep_base = .5 * (base * np.array([.299, .587, .114])).sum(-1, keepdims=True) + np.array([0, 0, 8])
    sleep_slope = np.repeat(.5 * (slope * np.array([.299, .587, .114])).sum(-1, keepdims=True), 3, axis=-1)
    base = np.concatenate([base, sleep_base])
    slope = np.concatenate([slope, sleep_slope])
    sample = np.asarray(frame[21:28, 28:35], np.float32)[mask]
    light = np.clip(((sample-base)*slope).sum((1,2)) / np.maximum((slope*slope).sum((1,2)), 1e-8), .5, 1.)
    detail = classify(sample, base + light[:,None,None]*slope, list(range(10)))
    pose = detail['value']
    world = frame[:49].astype(float)
    # Renderer sleep tint is a world effect, applied before the independently colored HUD.
    tint_error = np.maximum(np.abs(world[...,0]-world[...,1]), np.abs(world[...,2]-world[...,0]-8))
    coherent_tint = bool(np.mean(tint_error <= 2) >= .95)
    detail.update(version=VERSION, opaque_pixels=int(mask.sum()), sleep_tint_fraction=float(np.mean(tint_error <= 2)))
    direction, sleeping = None, None
    if pose is not None and pose < 4:
        direction, sleeping = pose+1, False
    elif pose == 9 and coherent_tint:
        sleeping = True
    else:
        detail['reason'] = 'unreadable_or_corrupted_pose_or_sleep_tint'
    return dict(direction=direction, sleeping=sleeping, player_evidence=detail)
