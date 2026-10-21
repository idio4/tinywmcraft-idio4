"""Independent label files and confusion matrices for saved RGB; no reference labels for generated pixels."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ..evidence import dump_json, load_case
from ..visual import HUD, hud_patch


def confusion(records):
    """Rows are independently labeled truth, columns reader output, including explicit abstentions."""
    total = len(records)
    known = [r for r in records if r['truth'] is not None]
    accepted = [r for r in known if r['prediction'] is not None]
    correct = sum(str(r['truth']) == str(r['prediction']) for r in accepted)
    labels = sorted({str(r['truth']) for r in known} | {str(r['prediction']) for r in accepted})
    matrix = {label: {column: 0 for column in labels + ['unknown']} for label in labels}
    for r in known:
        matrix[str(r['truth'])]['unknown' if r['prediction'] is None else str(r['prediction'])] += 1
    return {'total': total, 'human_or_reference_readable': len(known), 'accepted': len(accepted),
        'correct': correct, 'accepted_accuracy': correct / len(accepted) if accepted else None,
        'coverage_readable': len(accepted) / len(known) if known else None,
        'coverage_all': sum(r['prediction'] is not None for r in records) / total if total else None,
        'unreadable_accepted': sum(r['truth'] is None and r['prediction'] is not None for r in records),
        'labels': labels, 'confusion_matrix': matrix}


def export_review(evidence, out):
    """Small stratified sample; IDs omit expected outcomes and reader predictions to reduce anchoring."""
    evidence, out = Path(evidence), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    # Selected mechanics/fields, not selected based on reader success or expected correctness.
    selection = [
        ('gather_wood','wood'), ('gather_wood','tile'), ('gather_stone','stone'), ('gather_stone','tile'),
        ('gather_empty','wood'), ('gather_stone_no_tool','tile'),
        ('craft_wood_pickaxe','wood'), ('craft_wood_pickaxe','wood_pickaxe'),
        ('craft_stone_pickaxe','stone'), ('craft_stone_pickaxe','stone_pickaxe'),
        ('craft_iron_pickaxe','iron'), ('craft_iron_pickaxe','iron_pickaxe'),
        ('craft_iron_missing_resources','iron_pickaxe'), ('craft_no_table','wood'),
        ('craft_iron_no_furnace','coal'), ('craft_progression','wood'),
        ('craft_progression','stone_pickaxe'), ('consumed_resource_stays_gone','wood'),
        ('place_table','tile'), ('place_furnace','tile'), ('place_table_missing_resources','tile'),
        ('place_table_blocked','tile'), ('take_damage','health'), ('death','health'),
    ]
    samples = []
    available = json.loads((evidence/'manifest.json').read_text())['cases']
    for name, field in selection:
        candidates = [case for case in available if case.startswith(name+'_s')]
        if not candidates:
            continue
        i = len(samples)
        case = evidence / candidates[0]
        meta, arrays = load_case(case)
        t = meta['spec']['milestone'] - 1
        frame = arrays['b0_generated'][t]
        crop = frame[21:28,35:42] if field == 'tile' else hud_patch(frame, field)
        sample_id = f'R{i+1:02}'
        Image.fromarray(crop).save(out / f'{sample_id}.png')
        samples.append({'id': sample_id, 'case': case.name, 'branch': 0, 'frame_index': t,
                        'field': field, 'crop': f'{sample_id}.png', 'label': None, 'item_label': None})
    if not samples:
        raise ValueError('No reviewable cases in evidence')
    width, height = 960, ((len(samples)+3)//4) * 185
    sheet = Image.new('RGB', (width, height), (235,235,235)); draw = ImageDraw.Draw(sheet)
    for i, sample in enumerate(samples):
        x, y = (i % 4) * 240, (i // 4) * 185
        draw.text((x+12,y+8), f"{sample['id']}  {sample['field']}", fill='black')
        crop = Image.open(out / sample['crop']).resize((126,126), Image.Resampling.NEAREST)
        sheet.paste(crop,(x+12,y+32))
        draw.text((x+12,y+163),'value/type: __________',fill='black')
    sheet.save(out / 'review.png')
    dump_json(out / 'labels.json', {'schema_version':1,'evidence':str(evidence.resolve()),
        'verification':'pending_human_review','reviewer':None,'samples':samples,
        'instructions':'Label visible count and icon for HUD; tile type for tiles. Blank HUD slot=0/empty. Use null for ambiguous, never use reference state/action to fill a label.'})
    return out / 'review.png'


def validate_labels(reader, label_path, out):
    labels = json.loads(Path(label_path).read_text())
    records = {'digits': [], 'items': [], 'tiles': []}
    cache = {}
    for sample in labels['samples']:
        key = sample['case']
        if key not in cache:
            _, cache[key] = load_case(Path(labels['evidence']) / key)
        frame = cache[key][f"b{sample['branch']}_generated"][sample['frame_index']]
        read = reader.read(frame)
        group = 'tiles' if sample['field'] == 'tile' else 'digits'
        prediction = read['tiles']['3,5'] if group == 'tiles' else read[sample['field']]
        records[group].append({'id':sample['id'],'truth':sample['label'],'prediction':prediction})
        if group == 'digits':
            records['items'].append({'id':sample['id'],'truth':sample['item_label'],
                                     'prediction':read['items'][sample['field']]})
    report = {'verification': labels['verification'], 'reviewer': labels.get('reviewer'),
              'reader':reader.profile, 'groups':{g:confusion(rows) for g,rows in records.items()},'records':records,
              'limitations':'Small stratified pilot from one checkpoint; not a population accuracy guarantee.'}
    dump_json(out, report)
    csv_path=Path(out).with_name(Path(out).stem+'_confusion.csv')
    with csv_path.open('w',newline='') as file:
        writer=csv.DictWriter(file,['group','truth','prediction','count']);writer.writeheader()
        for group, values in report['groups'].items():
            for truth, columns in values['confusion_matrix'].items():
                for prediction,count in columns.items():
                    writer.writerow(dict(group=group,truth=truth,prediction=prediction,count=count))
    return report
