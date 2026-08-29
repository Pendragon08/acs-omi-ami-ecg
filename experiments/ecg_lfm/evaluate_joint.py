"""Record-level evaluation for corrected ECG-LFM joint checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from models.ecg_lfm_adapter import load_finetuned_model  # noqa: E402
from experiments.acs_joint_task import (  # noqa: E402
    ACS_CLASSES, LABELS, acs_metrics, acs_subtype_metrics, binary_metrics,
    encode_targets, save_confusion_figures, save_curves, save_roc_pr_curves,
    select_omi_threshold, sha256,
)


def _aggregate(logits: np.ndarray, indices: np.ndarray):
    unique = np.unique(indices); rows=[]
    for idx in unique:
        selected = logits[indices == idx]
        if len(selected) != 2: raise ValueError(f"record {idx} has {len(selected)} segments, expected 2")
        rows.append(selected.mean(0))
    return unique, np.asarray(rows, np.float32)


def infer_split(checkpoint: Path, manifest_dir: Path, split: str, segment_csv: Path, device: torch.device, workers: int, batch: int):
    model, cfg, task = load_finetuned_model(checkpoint, manifest_dir)
    model.eval().to(device); task.load_dataset(split, combine=False, epoch=1, task_cfg=cfg.task, label=False, shuffle=False); dataset = task.dataset(split)
    itr = task.get_batch_iterator(dataset=dataset, max_tokens=None, max_signals=batch, ignore_invalid_inputs=False, required_batch_size_multiple=1, seed=2026, num_shards=1, shard_id=0, num_workers=workers, data_buffer_size=0).next_epoch_itr(shuffle=False)
    logits = np.full((len(dataset), 5), np.nan, np.float32)
    with torch.no_grad():
        for sample in itr:
            if not sample: continue
            ids = sample['id'].cpu().numpy(); sample = {k: v for k, v in sample.items()} if device.type == 'cpu' else sample
            from fairseq_signals.utils import utils
            sample = utils.move_to_cuda(sample) if device.type == 'cuda' else sample
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                out = model(**sample['net_input']); batch_logits = model.get_logits(out).float().cpu().numpy()
            logits[ids] = batch_logits
    del model, task
    if not np.isfinite(logits).all(): raise RuntimeError(f"inference did not fill every {split} manifest row")
    with segment_csv.open(encoding='utf-8', newline='') as f: rows=list(csv.DictReader(f))
    if len(rows) != len(logits): raise ValueError("segment index and model output lengths differ")
    positions=np.asarray([int(r['manifest_position']) for r in rows]);
    if not np.array_equal(positions, np.arange(len(rows))): raise ValueError("segment CSV is not in manifest order")
    indices=np.asarray([int(r['idx']) for r in rows]); unique, rec_logits=_aggregate(logits,indices); p_acs=torch.softmax(torch.from_numpy(rec_logits[:,:4]),1).numpy(); p_omi=torch.sigmoid(torch.from_numpy(rec_logits[:,4])).numpy(); return unique,p_acs,p_omi


def read_history(run_root: Path, run_name: str) -> list[list[dict]]:
    histories=[]
    for fold in range(5):
        path=run_root/run_name/f'fold{fold}'/'fairseq_train.jsonl'; rows=[]
        if path.exists():
            for line in path.read_text(errors='ignore').splitlines():
                try:
                    item=json.loads(line)
                    if 'epoch' in item: rows.append({'epoch':item.get('epoch'),'train_total_loss':item.get('loss',item.get('train_loss',np.nan)),'val_total_loss':item.get('valid_loss',item.get('val_loss',np.nan)),'learning_rate':item.get('lr',item.get('learning_rate',np.nan))})
                except Exception: pass
        histories.append(rows)
    return histories


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run_name',required=True); p.add_argument('--run_root',default='results/corrected_ecg_lfm_joint'); p.add_argument('--manifest_root',default='manifests/ecg_lfm'); p.add_argument('--y_all',default='manifests/ecg_lfm/y_all.npy'); p.add_argument('--device',default='cuda:0'); p.add_argument('--workers',type=int,default=8); p.add_argument('--batch',type=int,default=32); args=p.parse_args(); device=torch.device(args.device if torch.cuda.is_available() else 'cpu'); run_root=ROOT/args.run_root; manifests=ROOT/args.manifest_root
    y_all=np.load(ROOT/args.y_all)
    with (manifests/'master_records.csv').open(encoding='utf-8',newline='') as f: master=list(csv.DictReader(f))
    train_indices=np.asarray([int(r['idx']) for r in master if r['cohort']=='train'],dtype=np.int64)
    if not np.array_equal(train_indices,np.arange(len(train_indices))): raise ValueError('train master indices must be contiguous 0..N-1')
    ysrc=y_all[train_indices]; acs,omi=encode_targets(ysrc); train_pos={int(i):j for j,i in enumerate(train_indices)}; oof_a=np.zeros((len(acs),4),np.float32); oof_o=np.zeros(len(omi),np.float32); fold_info=[]; seen=set(); started=time.time()
    master_by_idx={int(r['idx']):r for r in master}
    for fold in range(5):
        checkpoint=run_root/args.run_name/f'fold{fold}'/'checkpoint_best.pt'
        if not checkpoint.exists(): raise FileNotFoundError(f"best checkpoint missing (refusing arbitrary fallback): {checkpoint}")
        with (manifests/f'fold{fold}'/'valid_segments.csv').open(encoding='utf-8',newline='') as f: expected=np.unique(np.asarray([int(r['idx']) for r in csv.DictReader(f)],dtype=np.int64))
        unique,pa,po=infer_split(checkpoint,manifests/f'fold{fold}','valid',manifests/f'fold{fold}'/'valid_segments.csv',device,args.workers,args.batch)
        expected_from_master=np.asarray([int(idx) for idx,row in master_by_idx.items() if row['cohort']=='train' and row.get('validation_fold','')==str(fold)],dtype=np.int64)
        if not np.array_equal(unique, expected) or set(unique.tolist()) != set(expected_from_master.tolist()): raise ValueError(f"fold {fold} inference indices do not match master validation_fold")
        with (manifests/f'fold{fold}'/'train_segments.csv').open(encoding='utf-8',newline='') as f: train_patients={r['patient_id'] for r in csv.DictReader(f)}
        with (manifests/f'fold{fold}'/'valid_segments.csv').open(encoding='utf-8',newline='') as f: valid_patients={r['patient_id'] for r in csv.DictReader(f)}
        if train_patients.intersection(valid_patients): raise ValueError(f"patient leakage in fold {fold}")
        if seen.intersection(set(unique.tolist())): raise ValueError(f"duplicate OOF record indices across folds: fold {fold}")
        seen.update(unique.tolist())
        local=np.asarray([train_pos[int(i)] for i in unique],dtype=np.int64); oof_a[local]=pa; oof_o[local]=po; fold_info.append({'fold':fold,'n_records':int(len(unique)),'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'valid_segments_sha256':sha256(manifests/f'fold{fold}'/'valid_segments.csv')})
    if seen != set(train_indices.tolist()): raise RuntimeError('OOF validation manifests do not cover each training record exactly once')
    thresholds={}; details={}
    for m in ('fixed_0.5','max_f1','youden_j','sens90','sens95'): thresholds[m],details[m]=select_omi_threshold(omi,oof_o,m)
    pooled={m:{'acs':acs_metrics(acs,oof_a),'acs_subtype':acs_subtype_metrics(acs,oof_a),'omi':binary_metrics(omi,oof_o,t)} for m,t in thresholds.items()}; out=run_root/args.run_name/'summary'; out.mkdir(parents=True,exist_ok=True); np.savez_compressed(out/'oof_predictions.npz',acs=acs,omi=omi,p_acs=oof_a,p_omi=oof_o); (out/'threshold_analysis.json').write_text(json.dumps({'thresholds':thresholds,'details':details,'pooled_oof_metrics':pooled,'hard_metric_note':'pooled OOF threshold is selected and scored on the same OOF records; descriptive until nested calibration'},indent=2,ensure_ascii=False),encoding='utf-8'); manifest_hashes={};
    for fold in range(5):
        manifest_hashes[str(fold)]={'train_tsv':sha256(manifests/f'fold{fold}'/'train.tsv'),'valid_tsv':sha256(manifests/f'fold{fold}'/'valid.tsv'),'train_segments':sha256(manifests/f'fold{fold}'/'train_segments.csv'),'valid_segments':sha256(manifests/f'fold{fold}'/'valid_segments.csv'),'external_tsv':sha256(manifests/f'fold{fold}'/'external.tsv'),'external_segments':sha256(manifests/f'fold{fold}'/'external_segments.csv')}
    (out/'summary.json').write_text(json.dumps({'task':'ACS categorical OTHER/STEMI/NSTEMI/UA + OMI binary','run_name':args.run_name,'thresholds':thresholds,'pooled_oof_metrics':pooled,'folds':fold_info,'y_all_sha256':sha256(ROOT/args.y_all),'evaluator_sha256':sha256(Path(__file__)),'master_records_sha256':sha256(manifests/'master_records.csv'),'manifest_hashes':manifest_hashes,'elapsed_seconds':time.time()-started},indent=2,ensure_ascii=False),encoding='utf-8'); save_roc_pr_curves(out,acs,oof_a,omi,oof_o); cm_o=pooled['youden_j']['omi']; cm_omi=np.asarray([[cm_o['tn'],cm_o['fp']],[cm_o['fn'],cm_o['tp']]]); save_confusion_figures(out,np.asarray(pooled['youden_j']['acs']['confusion_matrix']),np.asarray(pooled['youden_j']['acs_subtype']['confusion_matrix']),cm_omi); save_curves(out,read_history(run_root,args.run_name),'Corrected ECG-LFM '+args.run_name)
    # External prediction and submission, without reading hidden labels.
    test_ids=[]; test_a=[]; test_o=[]
    external_unique = None; external_ids = None
    for fold in range(5):
        checkpoint=run_root/args.run_name/f'fold{fold}'/'checkpoint_best.pt'
        if not checkpoint.exists(): raise FileNotFoundError(f"best checkpoint missing: {checkpoint}")
        unique,pa,po=infer_split(checkpoint,manifests/f'fold{fold}','external',manifests/f'fold{fold}'/'external_segments.csv',device,args.workers,args.batch)
        with (manifests/f'fold{fold}'/'external_segments.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
        ids=np.asarray([next(r['record_id'] for r in rows if int(r['idx'])==int(i)) for i in unique],dtype=object)
        if external_unique is None: external_unique,external_ids=unique,ids
        elif not np.array_equal(unique,external_unique) or not np.array_equal(ids,external_ids): raise ValueError('external manifest order differs across folds')
        test_a.append(pa); test_o.append(po)
    arr=np.mean(test_a,0); oo=np.mean(test_o,0)
    with (ROOT/'data/CSV/test.csv').open(encoding='utf-8-sig',newline='') as f: test_ids=[(r.get('ecg_row_record') or '').strip() for r in csv.DictReader(f)]
    ext_by_id={str(record_id): i for i, record_id in enumerate(external_ids)}
    requested=[Path(x).stem for x in test_ids]
    if len(ext_by_id) != len(external_ids) or len(set(requested)) != len(requested) or set(requested) != set(ext_by_id):
        raise ValueError('external manifest record IDs do not uniquely match test.csv')
    order=np.asarray([ext_by_id[value] for value in requested]); arr=arr[order]; oo=oo[order]; pred=np.zeros((len(test_ids),4),np.int8); acp=arr.argmax(1)
    for k in (1,2,3): pred[:,k-1]=(acp==k).astype(np.int8)
    pred[:,3]=(oo>=thresholds['youden_j']).astype(np.int8); sub=run_root/args.run_name/'submission_corrected.csv';
    with sub.open('w',encoding='utf-8',newline='') as f: w=csv.writer(f); w.writerow(['ecg_row_record',*LABELS]); [w.writerow([rid,*row.tolist()]) for rid,row in zip(test_ids,pred)]
    (run_root/args.run_name/'external_summary.json').write_text(json.dumps({'external_labels_used':False,'omi_threshold':thresholds['youden_j'],'submission':str(sub),'submission_sha256':sha256(sub),'predicted_counts':pred.sum(0).tolist(),'external_segments_sha256':{str(fold):sha256(manifests/f'fold{fold}'/'external_segments.csv') for fold in range(5)}},indent=2),encoding='utf-8'); print(json.dumps({'run_name':args.run_name,'summary':str(out/'summary.json'),'submission':str(sub)},ensure_ascii=False))


if __name__=='__main__': main()
