"""CPU-only GIF rendering workers; deliberately independent of JAX and model code."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ACTION_NAMES = ['NOOP','LEFT','RIGHT','UP','DOWN','DO','SLEEP','PLACE_STONE',
                'PLACE_TABLE','PLACE_FURNACE','PLACE_PLANT','MAKE_WOOD_PICKAXE',
                'MAKE_STONE_PICKAXE','MAKE_IRON_PICKAXE','MAKE_WOOD_SWORD',
                'MAKE_STONE_SWORD','MAKE_IRON_SWORD']


def comparison_frames(meta, arrays, branch, status, scale=4, overall_status=None):
    """Include the last real conditioning frame, then all valid steps through terminal.

    Model padding beyond engine termination is deliberately excluded, never shown as a reset.
    """
    real, generated = arrays[f'b{branch}_reference'], arrays[f'b{branch}_generated']
    if len(generated) < len(real):
        raise ValueError('generated rollout shorter than reference')
    start = arrays['context'][-1]
    images = []
    size = 63 * scale
    extra = 16 if overall_status is not None else 0
    for t in range(len(real) + 1):
        canvas = Image.new('RGB', (max(660, 2 * size + 24), size + 108 + extra), '#eeeeee')
        draw = ImageDraw.Draw(canvas)
        action = 'NOOP (incoming marker)' if t == 0 else ACTION_NAMES[int(arrays[f'b{branch}_actions'][t-1])]
        terminal = t > 0 and bool(arrays[f'b{branch}_done_reference'][t-1])
        draw.text((10,5), meta['spec']['name'], fill='black')
        draw.text((10,21), f"engine seed={meta['seed']}  model seed={meta['model_seed']}  context={len(arrays['context'])}", fill='black')
        draw.text((10,37), f"t={t} action={action}  {'SELECTED' if branch == 0 else 'ALTERNATIVE'} branch", fill='black')
        draw.text((10,53), f"Branch mechanics at t={meta['spec']['milestone']}: {status.upper()}" + (' | REAL TERMINAL; stop here' if terminal else ''), fill='black')
        if overall_status is not None:
            label='Scenario overall'+(' ['+meta['score_scope']+']' if meta.get('score_scope') else '')+': '+overall_status.upper()
            if meta.get('ordinary_overall') is not None:label+=' | unconditional first: '+meta['ordinary_overall'].upper()
            draw.text((10,69),label,fill='black')
        for x, title, frame in [(8,'Real Craftax',start if t == 0 else real[t-1]),
                                 (size+16,'World model',start if t == 0 else generated[t-1])]:
            draw.text((x,75+extra),title,fill='black')
            canvas.paste(Image.fromarray(np.asarray(frame)).resize((size,size),Image.Resampling.NEAREST),(x,96+extra))
        images.append(canvas)
    return images


def render_case_gifs(case_dir, requests):
    """Render one case without importing JAX or loading a checkpoint.

    The parent already verified and copied the lossless trajectory.
    """
    with np.load(Path(case_dir)/'trajectory.npz',allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files}
    for filename,meta,branch,status,overall in requests:
        frames=comparison_frames(meta,arrays,branch,status,overall_status=overall)
        frames[0].save(Path(case_dir)/filename,save_all=True,append_images=frames[1:],
                       duration=250,loop=0,optimize=False,disposal=2)
    return len(requests)
