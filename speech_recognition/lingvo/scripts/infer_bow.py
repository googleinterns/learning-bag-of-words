import numpy as np
from tqdm import tqdm
import argparse
import os
import pickle as pkl
import scipy.optimize
import json

def solve_lp(A, b, c, lib='cvxopt'):
  if lib == 'scipy':
    res = scipy.optimize.linprog(c, A, b, bounds=[-5, 5])
    x = res.x
  elif lib == 'cvxopt':
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    A, b, c = matrix(A), matrix(b), matrix(c)
    sol = solvers.lp(c, A, b)
    x = sol['x']
    if x is not None:
      ret = A * x
      print(ret[0], np.min(ret[1:]), np.max(ret[1:]), np.count_nonzero(np.array(ret[1:]) <= 0))
      if args.lp_condition == 1:
        return np.count_nonzero(np.array(ret[1:]) <= 0) > 0.5 * len(ret)
      elif args.lp_condition == 2:
        return ret[0] < -0.5 and np.max(ret[1:]) < 1e-3
    return x is not None
  if x is None:
    return False
  ret = A @ x
  if np.all(ret <= b):
    return True
  return False

def solve_perceptron(X, y, fit_intercept=True, max_iter=1000, tol=1e-3, eta0=1.):
  from sklearn.linear_model import Perceptron
  clf = Perceptron(fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, eta0=eta0)
  clf.fit(X, y)
  if not fit_intercept:
    print(clf.score(X, y))
  if clf.score(X, y) == 1.0:
    return True
  return False

def bag_of_words(A, gt_k=None, epsilon=1e-8, method='perceptron'):
  m, n = np.shape(A)
  B, s, C = np.linalg.svd(A, full_matrices=False)
  # pred_k = np.linalg.matrix_rank(A)
  pred_k = int(np.sum(s > 1e-5))
  k = gt_k or pred_k
  print("Predicted length of target sequence:", pred_k)
  print("Finding SVD of W...")
  print("Eigenvalues:", s[:k])
  print("The rest:", s[k:])
  C = C[:k, :].astype(np.double)
  # C = C / np.linalg.norm(C, axis=0).astype(np.double)

  # Find x: x @ C has only one positive element
  bow = []
  for i in tqdm(range(n)):
    indices = [j for j in range(n) if j != i]
    np.random.shuffle(indices)
    if method == 'lp':
      if solve_lp(
              A=np.concatenate([C[:, i:i + 1], -C[:, indices]], 1).transpose(),
              b=np.array([-epsilon] + np.zeros(n - 1).tolist()),
              c=np.array(C[:, i:i + 1])
      ):
        bow.append(i)
    elif method == 'perceptron':
      if solve_perceptron(
        X=np.concatenate([C[:, i:i + 1], C[:, indices[:args.perceptron_sample_size]]], 1).transpose(),
        y=np.array([1 if j == 0 else -1 for j in range(args.perceptron_sample_size + 1)]),
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3
      ):
        bow.append(i)
        print("Added %d" % i)
  print(bow)

  if not args.relax:
    ret_bow = []
    for i in bow:
      if args.second_round == 'lp':
        indices = [j for j in range(n) if j != i]
        D = np.concatenate([C[:, i:i + 1], C[:, indices]], 1)
        indices2 = np.argsort(np.linalg.norm(D[:, 1:], axis=0))[-args.second_round_sample_size:]
        A = np.concatenate([D[:, 0:1], -D[:, 1 + indices2]], 1).transpose()
        # print(np.sort(np.linalg.norm(A, axis=1)))
        print("Token %d: %.4f" % (i, np.linalg.norm(C[:, i])))
        if solve_lp(
                A=A,
                b=np.array([-epsilon] + [0] * args.second_round_sample_size),
                c=np.array(C[:, i:i + 1])
        ):
          ret_bow.append(i)
          print("Selected %d" % i)
      elif args.second_round == 'perceptron':
        indices = [j for j in range(n) if j != i]
        indices.sort(key=lambda id: np.linalg.norm(C[:, id]), reverse=True)
        if solve_perceptron(
                X=np.concatenate([C[:, i:i + 1], C[:, indices[:args.second_round_sample_size]]], 1).transpose(),
                y=np.array([1 if j == 0 else -1 for j in range(args.second_round_sample_size + 1)]),
                fit_intercept=True,
                max_iter=5000,
                tol=None
        ):
          ret_bow.append(i)
          print("Added %d" % i)
    return pred_k, ret_bow
  else:
    return pred_k, bow

def apply_dropout(W, rate=0.2, dropout_type='weight'):
  if dropout_type == 'weight':
    shape = np.shape(W)
    W = np.reshape(W, [-1])
    indices = np.random.choice(range(np.shape(W)[0]), int(rate * np.shape(W)[0]))
    W[indices] = 0
    W = np.reshape(W, shape)
    return W
  elif dropout_type == 'one':
    for i in range(inp_dim):
      W[i, np.random.choice(range(num_classes))] = 0
    return W

def apply_noise(W, nm=0.2):
  W = W / np.linalg.norm(W)
  W = W + np.random.normal(scale=nm, size=np.shape(W))
  return W

if __name__ == "__main__":
  np.random.seed(0)

  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--uttid', type=str, help='utterance id')
  parser.add_argument('--tag', type=str, help='tag of reconstruction settings')
  parser.add_argument('--target_unit', type=str, help='word or char')
  parser.add_argument('--dropout', type=float, default=.0, help='dropout')
  parser.add_argument('--sign_sgd', type=bool, default=False, help='sign sgd')
  parser.add_argument('--dp_noise', type=float, default=.0, help='noise')
  parser.add_argument('--drop_grad', type=float, default=.0, help='drop gradient rate')
  parser.add_argument('--relax', type=bool, default=False, help='relax')
  parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon')
  parser.add_argument('--second_round', type=str, default='lp', help='lp or perceptron')
  parser.add_argument('--second_round_sample_size', type=int, default=5000, help='noise')
  parser.add_argument('--perceptron_sample_size', type=int, default=1000, help='noise')
  parser.add_argument('--lp_condition', type=int, default=1, help='')
  parser.add_argument('--fn', type=str, default='bow.txt', help='file name')

  args = parser.parse_args()

  ROOT = '/home/trungvd/lingvo-outputs/%s/%s' % (args.tag, args.uttid)
  with open(os.path.join(ROOT, "grads.pkl"), 'rb') as f:
    data = pkl.load(f)

  with open(os.path.join(ROOT, "info.json"), 'rb') as f:
    info = json.load(f)

  if args.target_unit == 'word':
    vocab_size = 16328
  else:
    vocab_size = 76
  A = data['grads'][-(3072 + 1) * vocab_size:-vocab_size]
  if args.drop_grad > 0:
    pos = int(len(A) * args.drop_grad)
    B = np.partition(np.abs(A), pos)
    A[np.abs(A) < B[pos]] = 0
    print("Drop percentage:", np.count_nonzero(A != 0) / len(A))
  A = np.reshape(A, [3072, vocab_size])

  if args.dropout > 0:
    A = apply_dropout(A, args.dropout)
  if args.dp_noise > 0:
    A = apply_noise(A, args.dp_noise)
  if args.sign_sgd:
    A = np.sign(A)

  _sum_add_1 = lambda ls: sum(ls) + len(ls) if type(ls) == list else ls + 1
  reconstructed_len = sum(_sum_add_1(l) for l in info['org_tgt_length'])
  print("Reconstruct with length %d" % reconstructed_len)

  target_len, target_bow = bag_of_words(A, gt_k=reconstructed_len, epsilon=args.epsilon)

  print("Results: Length: %d; BoW: %s" % (target_len, str(target_bow)))
  with open(os.path.join(ROOT, args.fn), 'w') as f:
    json.dump(dict(
      length=target_len,
      bow=target_bow
    ), f)