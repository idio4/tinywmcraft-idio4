"""Controlled fixtures for Craftax 1.5.0 Classic, using the unmodified engine."""
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import Action as A, BlockType as B
from craftax.craftax_classic.envs.craftax_state import Inventory
from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnvNoAutoReset
from craftax.craftax_classic.game_logic import calculate_light_level

CATEGORIES = ('movement', 'gathering', 'crafting', 'placement', 'combat', 'other')


def action_category(action):
    if 1 <= action <= 4:
        return 'movement'
    if action == 5:
        return 'gathering'  # combat DO is assigned by the scenario, not this action-only fallback
    if 7 <= action <= 10:
        return 'placement'
    if 11 <= action <= 16:
        return 'crafting'
    return 'other'


@dataclass(frozen=True)
class Scenario:
    name: str
    category: str
    valid: bool | None
    actions: tuple[int, ...]
    inventory: dict = field(default_factory=dict)
    front: int = B.GRASS.value
    table: bool = False
    furnace: bool = False
    zombie: str = ''
    # Exact post-intervention facts, checked against hidden reference state, never fed to the model.
    expected: dict = field(default_factory=dict)
    milestone: int = 1
    expected_tile: int | None = None  # screen tile (3,5) at milestone; otherwise not scored


def catalog():
    specs = [
        Scenario('move_free', 'movement', True, (2,), expected={'col_delta': 1}),
        Scenario('move_blocked', 'movement', False, (2,), front=B.STONE.value,
                 expected={'col_delta': 0}),
        Scenario('turn_loop', 'movement', True, (3, 1, 4, 2), milestone=4,
                 expected={'row_delta': 0, 'col_delta': 0, 'direction': 2}),
        Scenario('noop', 'other', None, (0,), expected={'row_delta': 0, 'col_delta': 0}),
        Scenario('gather_wood', 'gathering', True, (5,), front=B.TREE.value,
                 expected={'wood': 1}, expected_tile=B.GRASS.value),
        Scenario('gather_stone', 'gathering', True, (5,), front=B.STONE.value,
                 inventory={'wood_pickaxe': 1}, expected={'stone': 1}, expected_tile=B.PATH.value),
        Scenario('gather_empty', 'gathering', False, (5,), expected={'wood': 0, 'stone': 0}),
        Scenario('gather_stone_no_tool', 'gathering', False, (5,), front=B.STONE.value,
                 expected={'stone': 0}, expected_tile=B.STONE.value),
    ]
    for material, action, ingredients in [
        ('wood', 11, {'wood': 1}), ('stone', 12, {'wood': 1, 'stone': 1}),
        ('iron', 13, {'wood': 1, 'stone': 1, 'iron': 1, 'coal': 1}),
    ]:
        specs.append(Scenario(f'craft_{material}_pickaxe', 'crafting', True, (action,),
            inventory=ingredients, table=True, furnace=material == 'iron',
            expected={**{k: 0 for k in ingredients}, f'{material}_pickaxe': 1}))
        specs.append(Scenario(f'craft_{material}_missing_resources', 'crafting', False, (action,),
            table=True, furnace=material == 'iron', expected={f'{material}_pickaxe': 0}))
    specs += [
        Scenario('craft_no_table', 'crafting', False, (11,), inventory={'wood': 1},
                 expected={'wood': 1, 'wood_pickaxe': 0}),
        Scenario('craft_iron_no_furnace', 'crafting', False, (13,),
                 inventory={'wood': 1, 'stone': 1, 'coal': 1, 'iron': 1}, table=True,
                 expected={'wood': 1, 'stone': 1, 'coal': 1, 'iron': 1, 'iron_pickaxe': 0}),
        Scenario('place_table', 'placement', True, (8,), inventory={'wood': 2},
                 expected={'wood': 0}, expected_tile=B.CRAFTING_TABLE.value),
        Scenario('place_furnace', 'placement', True, (9,), inventory={'stone': 1},
                 expected={'stone': 0}, expected_tile=B.FURNACE.value),
        Scenario('place_table_missing_resources', 'placement', False, (8,),
                 expected={'wood': 0}, expected_tile=B.GRASS.value),
        Scenario('place_table_blocked', 'placement', False, (8,), inventory={'wood': 2},
                 front=B.STONE.value, expected={'wood': 2}, expected_tile=B.STONE.value),
        Scenario('attack_zombie', 'combat', True, (5,), zombie='attack',
                 inventory={'iron_sword': 1}, expected={'zombies': 0, 'health': 9}),
        Scenario('take_damage', 'combat', True, (0,), zombie='damage', expected={'health': 7}),
        Scenario('death', 'combat', True, (0,), zombie='death', expected={'health': 0, 'done': True}),
        Scenario('sleep', 'other', True, (6,), expected={'sleeping': True}),
        Scenario('craft_progression', 'crafting', True, (11, 12, 13),
                 inventory={'wood': 3, 'stone': 2, 'coal': 1, 'iron': 1}, table=True, furnace=True,
                 milestone=3, expected={'wood': 0, 'stone': 0, 'coal': 0, 'iron': 0,
                                         'wood_pickaxe': 1, 'stone_pickaxe': 1, 'iron_pickaxe': 1}),
        Scenario('consumed_resource_stays_gone', 'gathering', True, (5, 5, 0, 5),
                 front=B.TREE.value, milestone=4, expected={'wood': 1}, expected_tile=B.GRASS.value),
        # Row excursion takes the created table out of the 7-row viewport; returns at step 13.
        Scenario('table_leave_return', 'placement', True, (8,) + (3,) * 6 + (4,) * 6,
                 inventory={'wood': 2}, milestone=13, expected={'wood': 0, 'row_delta': 0},
                 expected_tile=B.CRAFTING_TABLE.value),
    ]
    return specs


class Engine:
    def __init__(self):
        self.env = CraftaxClassicPixelsEnvNoAutoReset()
        # Disable unsolicited spawns only; all movement, crafting, reward and death rules are unchanged.
        self.params = self.env.default_params.replace(spawn_cow_chance=0., spawn_zombie_base_chance=0.,
            spawn_zombie_night_chance=0., spawn_skeleton_chance=0.)
        self.reset = jax.jit(lambda key: self.env.reset(key, self.params))
        self.step = jax.jit(lambda key, state, action: self.env.step(key, state, action, self.params))
        self.render = jax.jit(self.env.get_obs)

    def construct(self, spec, seed, context):
        _, state = self.reset(jax.random.PRNGKey(seed))
        center = jnp.array([32, 32], jnp.int32)
        grid = jnp.full_like(state.map, B.GRASS.value)
        # Asymmetric landmarks make camera motion visible without obstructing the controlled arena.
        grid = grid.at[29, 29].set(B.TREE.value).at[34, 35].set(B.STONE.value)
        grid = grid.at[32, 33].set(spec.front)
        if spec.name in ('move_free','move_blocked','turn_loop','noop','table_leave_return'):
            # Visible asymmetric landmarks continue along the return route. Keep the walking arena clear.
            rng = np.random.default_rng(np.random.SeedSequence([seed, 3817]))
            for row in range(23,37):
                for col in range(28,37):
                    if col==32 or (31<=row<=33 and 31<=col<=33):
                        continue
                    grid = grid.at[row,col].set(int(rng.choice([2,4,5,7,8,9,13],p=[.4,.15,.15,.1,.08,.06,.06])))
        if spec.table:
            grid = grid.at[33, 32].set(B.CRAFTING_TABLE.value)
        if spec.furnace:
            grid = grid.at[33, 31].set(B.FURNACE.value)
        changes = {}
        for name in ('zombies', 'cows', 'skeletons', 'arrows'):
            mobs = getattr(state, name)
            changes[name] = mobs.replace(mask=jnp.zeros_like(mobs.mask), health=jnp.zeros_like(mobs.health),
                attack_cooldown=jnp.zeros_like(mobs.attack_cooldown))
        state = state.replace(**changes, map=grid, mob_map=jnp.zeros_like(state.mob_map),
            player_position=center, player_direction=jnp.asarray(2, jnp.int32), inventory=Inventory(**spec.inventory),
            player_health=jnp.asarray(2 if spec.zombie == 'death' else 9, jnp.int32),
            player_energy=jnp.asarray(5 if spec.name == 'sleep' else 9, jnp.int32),
            growing_plants_mask=jnp.zeros_like(state.growing_plants_mask),
            light_level=calculate_light_level(0, self.params), timestep=jnp.asarray(0, jnp.int32))
        if spec.zombie:
            mobs = state.zombies
            # During context NOOPs the zombie cannot move or attack: surround it, leaving the player adjacent.
            grid = state.map.at[31, 33].set(B.STONE.value).at[33, 33].set(B.STONE.value)
            grid = grid.at[32, 34].set(B.STONE.value)
            mobs = mobs.replace(position=mobs.position.at[0].set(center + jnp.array([0, 1])),
                health=mobs.health.at[0].set(5), mask=mobs.mask.at[0].set(True),
                attack_cooldown=mobs.attack_cooldown.at[0].set(context - 1))
            state = state.replace(zombies=mobs, map=grid, mob_map=state.mob_map.at[32, 33].set(True))
        return state


def u8(obs):
    return np.asarray(jnp.clip(obs * 255., 0, 255).astype(jnp.uint8))


def facts(state, initial_position, done=False):
    out = {name: int(getattr(state.inventory, name)) for name in state.inventory.__dataclass_fields__}
    pos = np.asarray(state.player_position)
    out.update(health=int(state.player_health), row_delta=int(pos[0] - initial_position[0]),
               col_delta=int(pos[1] - initial_position[1]), direction=int(state.player_direction),
               zombies=int(np.asarray(state.zombies.mask).sum()), sleeping=bool(state.is_sleeping), done=bool(done))
    return out
