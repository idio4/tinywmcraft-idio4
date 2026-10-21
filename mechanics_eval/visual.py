"""Conservative, independent RGB reader. Ambiguous generated pixels are NEVER assigned reference labels."""
import numpy as np
from craftax.craftax_classic.constants import TEXTURES
from .scenarios import u8

HUD = {'health': (0, 0), 'sapling': (0, 4), 'wood': (0, 5), 'stone': (0, 6),
       'coal': (0, 7), 'iron': (0, 8), 'diamond': (1, 0), 'wood_pickaxe': (1, 1),
       'stone_pickaxe': (1, 2), 'iron_pickaxe': (1, 3), 'wood_sword': (1, 4),
       'stone_sword': (1, 5), 'iron_sword': (1, 6)}


def hud_patch(frame, name):
    row, col = HUD[name]
    return np.asarray(frame)[49 + 7 * row:56 + 7 * row, 7 * col:7 * col + 7]


class VisualReader:
    # Pixel tolerance is fixed, conservative and never fit to model output or leaderboard scores.
    tolerance = 2.

    def __init__(self, engine, calibration_state):
        self.hud_templates = {}
        for name in HUD:
            patches = []
            for value in range(10):
                state = (calibration_state.replace(player_health=value) if name == 'health' else
                         calibration_state.replace(inventory=calibration_state.inventory.replace(**{name: value})))
                patches.append(hud_patch(u8(engine.render(state)), name))
            self.hud_templates[name] = np.asarray(patches, np.float32)
        texture = np.asarray(TEXTURES[7]['block_textures'], np.float32)
        # The renderer applies this exact daylight tint for light >= .5 and for awake players.
        lum = (texture * np.array([.299, .587, .114])).sum(-1, keepdims=True)
        night = .5 * (.4 * texture + .6 * lum) + .5 * np.array([0, 16, 64])
        self.tile_base, self.tile_slope = night, texture - night

    def read(self, frame):
        frame = np.asarray(frame)
        if frame.shape != (63, 63, 3) or frame.dtype != np.uint8:
            raise ValueError('visual reader expects uint8 RGB (63,63,3)')
        out = {}
        for name, templates in self.hud_templates.items():
            patch = hud_patch(frame, name).astype(np.float32)
            residual = np.max(np.abs(templates - patch), axis=(1, 2, 3))
            accepted = np.flatnonzero(residual <= self.tolerance)
            out[name] = int(accepted[0]) if len(accepted) == 1 else None
        tiles = {}
        for row in range(7):
            for col in range(9):
                if (row, col) == (3, 4):  # player overlay, never infer the hidden tile
                    continue
                patch = frame[row * 7:row * 7 + 7, col * 7:col * 7 + 7].astype(np.float32)
                slope = self.tile_slope
                level = np.clip(((patch - self.tile_base) * slope).sum((1, 2, 3)) /
                                np.maximum((slope * slope).sum((1, 2, 3)), 1e-8), .5, 1.)
                pred = self.tile_base + level[:, None, None, None] * slope
                residual = np.max(np.abs(pred - patch), axis=(1, 2, 3))
                accepted = np.flatnonzero(residual <= self.tolerance)
                tiles[f'{row},{col}'] = int(accepted[0]) if len(accepted) == 1 else None
        out['tiles'] = tiles
        return out

    def validate(self, frames, labels, states):
        """Held-out reference trajectories, not calibration templates; count both errors and abstentions."""
        hud_ok = hud_n = hud_total = tile_ok = tile_n = tile_total = 0
        for frame, truth, state in zip(frames, labels, states):
            read = self.read(frame)
            for name in HUD:
                hud_total += 1
                if read[name] is not None:
                    hud_n += 1; hud_ok += int(read[name] == truth[name])
            pos = np.asarray(state.player_position)
            # Active mob tiles are occluded and excluded from clean-tile calibration accuracy.
            occluded = set()
            for name in ('zombies', 'cows', 'skeletons', 'arrows'):
                mobs = getattr(state, name)
                occluded.update(tuple(p) for p in np.asarray(mobs.position)[np.asarray(mobs.mask)])
            for key, value in read['tiles'].items():
                row, col = map(int, key.split(','))
                p = tuple(pos + np.array([row - 3, col - 4]))
                if p in occluded or bool(state.is_sleeping) or float(state.light_level) < .5:
                    continue
                tile_total += 1
                if value is not None:
                    tile_n += 1; tile_ok += int(value == int(state.map[p]))
        return {'hud_correct': hud_ok, 'hud_accepted': hud_n, 'hud_total': hud_total,
                'tile_correct': tile_ok, 'tile_accepted': tile_n, 'tile_total': tile_total}
