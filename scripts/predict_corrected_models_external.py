"""Create external submissions for corrected categorical-ACS models.

ACS predictions are obtained by softmax argmax; only OMI uses the pooled OOF
validation threshold.  No external labels are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import hmoe_kan_github as h  # noqa: E402
from experiments.acs_joint_model_comparison import build_model, logits  # noqa: E402
from experiments.acs_joint_task import ACS_CLASSES, LABELS, stem  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_test():
    arr = ROOT / "data" / "arrays"; p2 = np.load(arr / "P2_test.npy", mmap_mode="r"); sc = json.loads((arr / "scalers.json").read_text())['P2']; x = p2.astype(np.float32) * (float(sc['scale']) + 1e-8) + float(sc['median']); meta = np.load(arr / "meta_test.npz", allow_pickle=True); meta_ids = np.asarray([stem(x) for x in meta['record_ids']], object)
    with (ROOT / "data" / "CSV" / "test.csv").open(encoding="utf-8-sig", newline="") as f: ids = [(r.get('ecg_row_record') or '').strip() for r in csv.DictReader(f)]
    by_id = {str(x): i for i, x in enumerate(meta_ids)}
    requested = [stem(x) for x in ids]
    if len(by_id) != len(meta_ids) or len(set(requested)) != len(requested) or set(requested) != set(by_id):
        raise ValueError("test.csv identifiers must uniquely match meta_test identifiers")
    order = np.asarray([by_id[value] for value in requested]); return x[order], ids


def predict(model, x, device, batch):
    pa, po = [], []; model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out = logits(model, torch.from_numpy(np.asarray(x[i:i+batch], np.float32)).to(device)); pa.append(torch.softmax(out[:, :4], 1).cpu().numpy()); po.append(torch.sigmoid(out[:, 4]).cpu().numpy())
    return np.concatenate(pa), np.concatenate(po)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--models', default='InceptionTime,ResNet1D_Wang,LSTM,Transformer1D,ViT'); p.add_argument('--result_root', default='results/corrected_joint_model_comparison'); p.add_argument('--device', default='cuda:0'); p.add_argument('--batch_size', type=int, default=128); args = p.parse_args(); device = torch.device(args.device if torch.cuda.is_available() else 'cpu'); x, ids = load_test(); root = ROOT / args.result_root
    for name in [z.strip() for z in args.models.split(',') if z.strip()]:
        model_root = root / name; summary = json.loads((model_root / 'summary.json').read_text()); threshold = float(summary['thresholds']['youden_j']); pa_folds=[]; po_folds=[]
        for fold in range(5):
            sc = json.loads((model_root / f'fold{fold}' / 'scaler.json').read_text()); xt = h.apply_robust_scaler(x, sc); model = build_model(name).to(device); state = torch.load(model_root / f'fold{fold}' / 'best_model.pt', map_location=device); model.load_state_dict(state); pa, po = predict(model, xt, device, args.batch_size); pa_folds.append(pa); po_folds.append(po)
        pa=np.mean(pa_folds,0); po=np.mean(po_folds,0); ac=pa.argmax(1); pred=np.zeros((len(ids),4),np.int8)
        for k in (1,2,3): pred[:,k-1]=(ac==k).astype(np.int8)
        pred[:,3]=(po>=threshold).astype(np.int8); out=model_root/f'external_test_predictions_corrected.csv'; sub=model_root/f'submission_corrected_{name}.csv'
        with out.open('w',encoding='utf-8',newline='') as f: w=csv.writer(f); w.writerow(['ecg_row_record',*[f'p_acs_{n}' for n in ACS_CLASSES],'p_omi','acs_pred','omi_pred']); [w.writerow([rid,*a.tolist(),float(o),ACS_CLASSES[int(c)],int(op)]) for rid,a,o,c,op in zip(ids,pa,po,ac,pred[:,3])]
        with sub.open('w',encoding='utf-8',newline='') as f: w=csv.writer(f); w.writerow(['ecg_row_record',*LABELS]); [w.writerow([rid,*pr.tolist()]) for rid,pr in zip(ids,pred)]
        (model_root/'external_prediction_provenance_corrected.json').write_text(json.dumps({'task':'ACS categorical + OMI binary','threshold_omi':threshold,'external_labels_used':False,'n_external_records':len(ids),'submission':str(sub),'submission_sha256':sha256(sub),'predicted_counts':pred.sum(0).tolist()},indent=2),encoding='utf-8')
        print(json.dumps({'model':name,'submission':str(sub),'predicted_counts':pred.sum(0).tolist()}),flush=True)


if __name__=='__main__': main()
