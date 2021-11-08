"""Analyze reconstruction results."""

import numpy as np
from tqdm import tqdm
import argparse
import os
import sklearn.metrics
import json
import editdistance


def ensure_list(ls):
  if type(ls) == list:
    return ls
  else:
    return [ls]


def get_utts(tag, ids_fp):
  ROOT = '/home/trungvd/lingvo-outputs'
  utts = []
  with open(ids_fp) as f:
    lines = f.read().strip().split('\n')
    utt_ids = [l.split(' ')[0] for l in lines]
  for uttid in utt_ids:
    utt_root = os.path.join(ROOT, tag, uttid)
    if args.only_ignored and not os.path.exists(os.path.join(utt_root, 'ignore')):
      continue
    utt = {}
    utt['id'] = uttid
    utt['org_transcript'] = [' '.join(all_transcript[uid]) for uid in uttid.split(',')]
    utt['org_length'] = len(utt['org_transcript'])
    utt['org_bow_length'] = len(set(utt['org_transcript']))
    utt['has_bow'] = os.path.exists(os.path.join(utt_root, 'bow.txt'))
    if not utt['has_bow']:
      print("%s does not have bow" % utt['id'])
    try:
      with open(os.path.join(utt_root, 'bow.txt')) as f:
        data = json.load(f)
        utt['rec_bow'] = set([t for t in data['bow'] if t > 2])
        utt['rec_tgt_length'] = data['length'] - len(utt['org_transcript'])
      with open(os.path.join(utt_root, 'info.json')) as f:
        data = json.load(f)
      utt['org_src_length'] = data['org_src_length']
      data['org_tgt_length'] = ensure_list(data['org_tgt_length'])
      utt['org_tgt_length'] = [l for l in data['org_tgt_length']]
      utt['org_tgt_labels'] = [lbls[:length] for lbls, length in zip(data['org_tgt_labels'], utt['org_tgt_length'])]
      utt['org_bow'] = set()
      for lbls in utt['org_tgt_labels']:
        utt['org_bow'] |= set([l for l in lbls if l > 2])
      utt['org_tgt_str_labels'] = data['org_tgt_str_labels']
      # utt['rec_bow'] = set(data.get('rec_bow') or [])
      # if data['finished']:
      #   utt['rec_tgt_labels'] = data['rec_tgt_labels'][0]
      #   utt['rec_tgt_str_labels'] = data['rec_tgt_str_labels'][0][0]
      #   utt['num_steps'] = data['num_steps']
      #   utt['wer'] = editdistance.eval(utt['org_tgt_labels'][0], utt['rec_tgt_labels']) / len(utt['org_tgt_labels'])
      #   utt['bow_acc'] = set(utt['rec_bow']) == set(utt['org_bow'])
      #   utt['loss'] = data['loss'] if 'loss' in data else None
      if os.path.exists(os.path.join(utt_root, 'history')):
        with open(os.path.join(utt_root, 'history')) as f:
          data = f.read().strip().split('\n')
          utt['num_reconstructed_times'] = len([line for line in data if line[:8] == 'Finished'])
      else:
        utt['num_reconstructed_times'] = 0
    except FileNotFoundError:
      utt['org_tgt_length'] = None
      utt['org_tgt_labels'] = None
      utt['rec_bow'] = None
      utt['finished'] = False
      utt['wer'] = None
      utt['rec_tgt_labels'] = None
      utt['bow_acc'] = None
      utt['loss'] = None
      utt['num_reconstructed_times'] = None
    utts.append(utt)
  return utts
  # return [u for u in utts if u['org_tgt_length'] is not None and sum(u['org_tgt_length']) == u['rec_tgt_length']]

def report(utts):
    ret = {}
    ret['num_utts'] = len(utts)
    ret['num_bow'] = len([u for u in utts if u['has_bow']])
    # ret['num_finished'] = len([u for u in utts if u['finished']])
    has_bow_utts = [u for u in utts if u['rec_bow']]
    # finished_utts = [u for u in utts if u['finished']]
    ret['length_err'] = np.average(
      [1 - u['rec_tgt_length'] / sum(u['org_tgt_length']) for u in utts if u['org_tgt_length'] and u['rec_tgt_length']])
    y_true = []
    y_pred = []
    num_classes = 16328
    for utt in has_bow_utts:
      if utt['org_tgt_labels'] is None:
        continue
      y_true += [i in utt['org_bow'] for i in range(2, num_classes)]
      y_pred += [i in utt['rec_bow'] for i in range(2, num_classes)]
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    ret['bow_prec'] = tp / (tp + fp)
    ret['bow_recall'] = tp / (tp + fn)
    ret['bow_acc'] = (tp + tn) / (tp + tn + fp + fn)
    ret['bow_em'] = sum([u['rec_bow'] == u['org_bow'] for u in has_bow_utts]) / len(has_bow_utts)

    def wer(utts):
      dist = 0
      length = 0
      for utt in utts:
        dist += editdistance.eval(utt['org_tgt_labels'], utt['rec_tgt_labels'])
        length += len(utt['org_tgt_labels'])
      return dist / length if length > 0 else None

    # ret['transcript_norm_er'] = wer(finished_utts)
    # ret['transcript_er'] = np.average([u['wer'] for u in finished_utts])
    # ret['transcript_em'] = np.average([u['wer'] == 0. for u in finished_utts])
    return ret


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--tag', type=str, help='tag of reconstruction settings')
  parser.add_argument('--ids_fp', type=str, default='./data/test-sorted_less_50.txt', help='path to ids file')
  parser.add_argument('--only_ignored', type=bool, default=False, help='only count utt with ignore')
  parser.add_argument('--noise', type=float, default=.0, help='noise')

  args = parser.parse_args()

  lines = []
  with open('data/test-sorted_less_50.txt') as f:
    ls = f.read().split('\n')
    ls = [l.split(' ', 1) for l in ls if l != ""]
    lines += ls
  tokens = open('data/vocab.txt').read().strip().split('\n')
  tokens = [t.replace('▁', '') for t in tokens]
  token2id = {t: i for i, t in enumerate(tokens)}
  # all_transcript = {uid: [token2id[t.lower()] for t in transcript.split(' ')] for uid, transcript in lines}
  all_transcript = {uid: [t.lower() for t in transcript.split(' ')] for uid, transcript in lines}

  utts = get_utts(args.tag, args.ids_fp)
  report = report(utts)

  for key, val in report.items():
    print("%s:\t%s" % (key, str(val)))
