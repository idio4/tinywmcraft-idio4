"""CPU-only RGB measurements with separate digits/icons and renderer-derived disjoint support regions."""
import hashlib
from pathlib import Path
import numpy as np
from craftax.craftax_classic.constants import TEXTURES
from .visual import HUD, hud_patch, VisualReader
from .scenarios import u8

VERSION = 'cpu-glyph-icon-rms-v2.2-player-pose'
SUPPORT_FRACTION = .25  # Fixed before generated-image audit. A quarter of clean inter-class separation.


def classify(patch, templates, labels, mask=None, fraction=SUPPORT_FRACTION):
    """Nearest class, accepted only inside a class-specific separation-derived support region."""
    values = np.asarray(templates, np.float32)
    sample = np.asarray(patch, np.float32)
    if mask is not None:
        values, sample = values[:, mask], sample[mask]
    flat = values.reshape(len(values), -1)
    distances = np.sqrt(np.mean((flat - sample.reshape(1, -1)) ** 2, axis=1))
    inter = np.sqrt(np.mean((flat[:, None] - flat[None]) ** 2, axis=-1))
    np.fill_diagonal(inter, np.inf)
    separation = inter.min(axis=1)
    order = np.argsort(distances, kind='stable')
    winner, second = map(int, order[:2])
    radius = float(fraction * separation[winner])
    accepted = radius > 0 and distances[winner] <= radius
    detail = {'value': labels[winner] if accepted else None, 'candidate': labels[winner],
              'runner_up': labels[second], 'rms': float(distances[winner]),
              'runner_up_rms': float(distances[second]), 'support_radius': radius,
              'margin': float(distances[second] - distances[winner]),
              'accepted': bool(accepted), 'reason': 'inside_calibrated_support' if accepted else 'outside_calibrated_support'}
    return detail


class ReaderV2:
    """A frame is the only read-time input. No actions, rewards, hidden state or expected outcomes."""
    def __init__(self, engine, state):
        self.strict = VisualReader(engine, state)
        self.templates = self.strict.hud_templates
        self.digit_mask = np.zeros((7, 7), bool)
        self.digit_mask[2:6, 2:6] = True  # Actual Classic renderer number placement, not the item icon.
        self.icon_mask = ~self.digit_mask
        self.item_names = ['empty', *HUD]
        self.icons = np.stack([self.templates['wood'][0], *[self.templates[name][1] for name in HUD]])
        self.tile_labels = list(range(2, 17))  # Exclude invalid/debug/out-of-bounds aliases.
        self.base = self.strict.tile_base[self.tile_labels]
        self.slope = self.strict.tile_slope[self.tile_labels]
        # Renderer sprites are overlays, not evidence that the underlying terrain changed.
        self.overlays = {}
        textures = TEXTURES[7]
        blocks = np.asarray(textures['block_textures'], np.float32)[self.tile_labels]
        for entity in ('cow', 'zombie', 'skeleton'):
            alpha = np.asarray(textures[entity+'_texture_alpha'])
            composite = blocks * (1-alpha) + np.asarray(textures[entity+'_texture']) * alpha
            lum = (composite * np.array([.299, .587, .114])).sum(-1, keepdims=True)
            base = .5 * (.4 * composite + .6 * lum) + .5 * np.array([0, 16, 64])
            self.overlays[entity] = (base, composite-base, np.all(alpha == 0, axis=-1))
        self.profile = {'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'version': VERSION, 'support_fraction': SUPPORT_FRACTION,
                        'support_definition': 'RMS radius = 0.25 * nearest other clean class distance',
                        'template_sha256': hashlib.sha256(b''.join(v.tobytes() for v in self.templates.values())).hexdigest(),
                        'training': 'none', 'uses_actions_or_reference_labels': False}
        from . import player_reader, target_reader
        self.profile['player_reader_sha256'] = hashlib.sha256(Path(player_reader.__file__).read_bytes()).hexdigest()
        self.profile['target_reader_sha256'] = hashlib.sha256(Path(target_reader.__file__).read_bytes()).hexdigest()
        self.profile['entity_overlays'] = list(self.overlays)
        self.profile['overlay_template_sha256'] = hashlib.sha256(b''.join(
            array.tobytes() for values in self.overlays.values() for array in values)).hexdigest()

    def tile(self, patch):
        patch = np.asarray(patch, np.float32)
        level = np.clip(((patch - self.base) * self.slope).sum((1, 2, 3)) /
                        np.maximum((self.slope ** 2).sum((1, 2, 3)), 1e-8), .5, 1.)
        templates = self.base + level[:, None, None, None] * self.slope
        detail = classify(patch, templates, self.tile_labels)
        detail['fitted_daylight'] = float(level[self.tile_labels.index(detail['candidate'])])
        if not detail['accepted']:
            matches = []
            for entity, (base, slope, visible) in self.overlays.items():
                lighting = np.clip(((patch-base)*slope).sum((1,2,3)) /
                                   np.maximum((slope*slope).sum((1,2,3)), 1e-8), .5, 1.)
                composites = base + lighting[:,None,None,None]*slope
                overlay = classify(patch, composites, self.tile_labels)
                # Check terrain separation using only pixels actually visible around the sprite.
                terrain = classify(patch, composites, self.tile_labels, mask=visible)
                if overlay['accepted'] and terrain['accepted'] and overlay['value'] == terrain['value']:
                    matches.append(dict(terrain, entity=entity, visible_pixels=int(visible.sum()),
                                        overlay_fit=overlay, reason='terrain_visible_around_entity',
                                        fitted_daylight=float(lighting[self.tile_labels.index(terrain['value'])])))
            if len(matches) == 1:
                return matches[0]
            if matches:
                detail['reason'] = 'ambiguous_entity_overlay'
        return detail

    def read(self, frame):
        frame = np.asarray(frame)
        if frame.shape != (63, 63, 3) or frame.dtype != np.uint8:
            raise ValueError('Reader requires RGB uint8 (63,63,3)')
        if np.all(frame == frame[0,0]):
            # A missing/constant video frame is not evidence of a zero-valued inventory.
            rejected = {'value':None,'candidate':None,'runner_up':None,'rms':None,
                        'runner_up_rms':None,'support_radius':None,'margin':None,
                        'accepted':False,'reason':'uniform_frame'}
            tile_keys = [f'{r},{c}' for r in range(7) for c in range(9) if (r,c)!=(3,4)]
            return {**{k:None for k in HUD},'items':{k:None for k in HUD},
                    'tiles':{k:None for k in tile_keys},
                    'measurements':{**{k:{'digit':dict(rejected),'item':dict(rejected)} for k in HUD},
                                    **{'tile:'+k:dict(rejected) for k in tile_keys}}}
        values, details, items, tiles = {}, {}, {}, {}
        for field in HUD:
            patch = hud_patch(frame, field)
            digit = classify(patch, self.templates[field], list(range(10)), self.digit_mask)
            item = classify(patch, self.icons, self.item_names, self.icon_mask)
            value = digit['value']
            if value is not None:
                expected_item = 'empty' if value == 0 else field
                if item['value'] != expected_item:
                    digit = dict(digit, value=None, accepted=False,
                                 reason='unreadable_item' if item['value'] is None else 'item_digit_inconsistent')
            values[field] = digit['value']
            items[field] = item['value']
            details[field] = {'digit': digit, 'item': item}
        for row in range(7):
            for col in range(9):
                if (row, col) != (3, 4):
                    key = f'{row},{col}'
                    reading = self.tile(frame[row*7:row*7+7, col*7:col*7+7])
                    tiles[key] = reading['value']
                    details['tile:' + key] = reading
        from .player_reader import read_player
        values.update(read_player(frame))
        values.update(tiles=tiles, items=items, measurements=details)
        return values
