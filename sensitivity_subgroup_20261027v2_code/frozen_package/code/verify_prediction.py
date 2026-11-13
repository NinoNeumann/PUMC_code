"""Reject incomplete or synthetic prediction artifacts before formal evaluation."""
from pathlib import Path
import json
import sys
from portable_hash import verify_complete

def verify_prediction(directory):
    p=Path(directory)
    d=json.loads((p/'COMPLETE.json').read_text(encoding='utf-8'))
    if d.get('status')!='complete' or d.get('production') is not True or d.get('replay_passed') is not True:
        raise ValueError('A formal, replay-verified COMPLETE marker is required')
    if d.get('genuine_views') not in (1,20):raise ValueError('Invalid real imputation view count')
    branch=d.get('branch')
    allowed={'complete_case','egfr_observed','renal_observed','known_subtype','no_egpa'}
    if branch not in allowed or d['genuine_views']!=(1 if branch=='complete_case' else 20):
        raise ValueError('Genuine view count does not match the sensitivity branch')
    if p.parent.parent.name!=branch:
        raise ValueError('Completion branch does not match its directory')
    if not {'predictions_all20.csv','manifest.json'}<=set(d.get('files',{})):
        raise ValueError('Predictions and manifest must both have signed hashes')
    verify_complete(p)
    return d

if __name__=='__main__':
    try:verify_prediction(sys.argv[1]);print('FORMAL_PREDICTION_VERIFIED')
    except Exception as e:print(str(e),file=sys.stderr);sys.exit(1)
