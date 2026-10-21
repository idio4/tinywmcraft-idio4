"""Small mechanics suite and controlled initial-state variation; reference engine stays unchanged."""
from dataclasses import replace
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import BlockType
from craftax.craftax_classic.game_logic import calculate_light_level
from .scenarios import Engine, catalog

GROUPS = {
    'valid_crafting': ('craft_wood_pickaxe', 'craft_stone_pickaxe', 'craft_iron_pickaxe'),
    'invalid_crafting': ('craft_wood_missing_resources', 'craft_stone_missing_resources',
                         'craft_iron_missing_resources', 'craft_no_table', 'craft_iron_no_furnace'),
    'resource_collection': ('gather_wood', 'gather_stone'),
    'invalid_interaction': ('gather_empty', 'gather_stone_no_tool'),
    'resource_accounting': ('craft_progression', 'consumed_resource_stays_gone'),
}


def category(name):
    return next((group for group, names in GROUPS.items() if name in names), 'diagnostic_other')


def variants(spec, variant):
    """Preserve each intervention's feasibility while varying surplus and irrelevant initial resources."""
    if spec.name == 'gather_empty':
        # Grass DO can legitimately yield a sapling (10%); path has no collectable resource.
        spec = replace(spec, front=BlockType.PATH.value, expected_tile=BlockType.PATH.value)
    if not variant:
        return spec
    inventory, expected = dict(spec.inventory), dict(spec.expected)
    extra = 1 + (variant - 1) % 2
    if spec.name in GROUPS['valid_crafting'] + ('craft_progression', 'craft_iron_no_furnace', 'craft_no_table'):
        for key in ('wood', 'stone', 'iron', 'coal'):
            if key in inventory:
                inventory[key] += extra
                expected[key] += extra
    elif spec.name in ('gather_wood', 'gather_stone', 'consumed_resource_stays_gone', 'gather_stone_no_tool'):
        key = 'stone' if 'stone' in spec.name else 'wood'
        inventory[key] = extra
        expected[key] += extra
    elif spec.name == 'craft_stone_missing_resources':
        inventory['wood' if variant % 2 else 'stone'] = 1
        expected.update(inventory)
    elif spec.name == 'craft_iron_missing_resources':
        inventory.update(wood=1, stone=1, iron=1, coal=1)
        inventory['coal' if variant % 2 else 'stone'] = 0
        expected.update(inventory)
    inventory['sapling'] = extra
    return replace(spec, inventory=inventory, expected=expected)


class VariedEngine(Engine):
    def construct(self, spec, seed, context):
        state = super().construct(spec, seed, context)
        # Seeded peripheral backgrounds; preserve the central interaction/station arena.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1771]))
        tiles = [BlockType.GRASS.value, BlockType.SAND.value, BlockType.PATH.value,
                 BlockType.TREE.value, BlockType.STONE.value]
        grid = state.map
        for row in (29, 30, 34, 35):
            for col in range(28, 37):
                grid = grid.at[row, col].set(int(rng.choice(tiles, p=[.65,.15,.1,.05,.05])))
        timestep = int(seed % 3) * 12
        return state.replace(map=grid, timestep=jnp.asarray(timestep, jnp.int32),
                             light_level=calculate_light_level(timestep, self.params))


def suite(name):
    return [s for s in catalog() if name == 'existing' or category(s.name) != 'diagnostic_other']
