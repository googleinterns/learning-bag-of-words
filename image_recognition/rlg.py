import argparse
import json
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, default='ResNet50', help='Name of a tf.keras.applications model, default: ResNet50')
parser.add_argument('--checkpoint', type=str, default='imagenet', help='Checkpoint of the model. Can be a checkpoint name for tf.keras.applications models (e.g. imagenet) or a path')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--num_batches', type=int, default=100, help='num batches')
parser.add_argument('--signsgd', type=bool, default=False, help='apply sign-sgd')
parser.add_argument('--dropgrad', type=float, default=0, help='gradient dropping')
parser.add_argument('--output_dir', type=str, default='./outputs', help="Root dir.")
args = parser.parse_args()


def solve_lp(A, b, c):
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    np.random.seed(None)
    for t in range(1):
        A, b, c = matrix(A), matrix(b), matrix(c)
        sol = solvers.lp(c, A, b)
        x = sol['x']
        if x is not None:
            ret = A * x
            # return np.count_nonzero(np.array(ret[1:]) <= 0) > 0.9 * len(ret)
            if ret[0] < -0.1 and np.max(ret[1:]) < 1e-2 and np.count_nonzero(np.array(ret[1:]) <= 0) > 0.5 * len(ret):
                return True
    return False
  
def solve_perceptron(X, y, fit_intercept=True, max_iter=1000, tol=1e-3, eta0=1.):
    from sklearn.linear_model import Perceptron
    clf = Perceptron(fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, eta0=eta0)
    clf.fit(X, y)
    if not fit_intercept:
        pass
    if clf.score(X, y) > 0.9:
        return True
    return False

def infer_labels(A, gt_k=None, epsilon=1e-8, method='perceptron'):
    m, n = np.shape(A)
    B, s, C = np.linalg.svd(A, full_matrices=False)
    pred_k = np.linalg.matrix_rank(A)
    k = gt_k or pred_k
    print("Predicted length of target sequence:", pred_k)
    print("Finding SVD of W...")
    print(s[:gt_k])
    print(s[gt_k])
    C = C[:k, :].astype(np.double)

    # Find x: x @ C has only one positive element
    # Filter possible labels using perceptron algorithm
    bow = []
    if args.model == "ResNet50":
        bow = np.reshape(np.where(np.min(A, 0) < 0), -1).tolist()
    for i in range(n):
        if i in bow:
            continue
        indices = [j for j in range(n) if j != i]
        np.random.shuffle(indices)
        if solve_perceptron(
                X=np.concatenate([C[:, i:i + 1], C[:, indices[:999]]], 1).transpose(),
                y=np.array([1 if j == 0 else -1 for j in range(1000)]),
                fit_intercept=True,
                max_iter=1000,
                tol=1e-3
        ):
            bow.append(i)
  
    # Get the final set with linear programming
    ret_bow = []
    for i in bow:
        if i in ret_bow:
            continue
        indices = [j for j in range(n) if j != i]
        D = np.concatenate([C[:, i:i + 1], C[:, indices]], 1)
        indices2 = np.argsort(np.linalg.norm(D[:, 1:], axis=0))[-199:]
        A = np.concatenate([D[:, 0:1], -D[:, 1 + indices2]], 1).transpose()
        if solve_lp(
                A=A,
                b=np.array([-epsilon] + [0] * len(indices2)),
                c=np.array(C[:, i:i + 1])
        ):
            ret_bow.append(i)
    return pred_k, ret_bow

CACHE_DIR = os.path.join(
    args.output_dir,
    "%s_%s_bs_%d" % (args.model, 'random' if args.checkpoint is None else args.checkpoint, args.batch_size))

results = []
for i in tqdm(range(args.num_batches)):
    grad_path = os.path.join(CACHE_DIR, "grad-%d.npy" % (i))
    label_path = os.path.join(CACHE_DIR, "label-%d.npy" % (i))
    
    grads = np.load(grad_path)
    labels = np.load(label_path)
    
    if args.signsgd:
        grads = np.sign(grads)
    if args.dropgrad > 0:
        A = np.reshape(grads, -1)
        pos = int(len(A) * args.dropgrad)
        B = np.partition(np.abs(A), pos)
        A[np.abs(A) < B[pos]] = 0
        grads = np.reshape(A, grads.shape)
    
    gt_labels = set(labels.tolist())
    _, pred_labels = infer_labels(grads, gt_k=args.batch_size, epsilon=1e-10)
    
    print("In gt, not in pr:", [i for i in gt_labels if i not in pred_labels])
    print("In pr, not in gt:", [i for i in pred_labels if i not in gt_labels])
    
    results.append({
        'ref': sorted(list(gt_labels)),
        'hyp': sorted(list(pred_labels))
    })
    
data = json.dumps(results)

fn = 'results_svd'

if args.signsgd:
    fn += '_signsgd'
if args.dropgrad > 0:
    fn += '_dropgrad_%f' % args.dropgrad

with open(os.path.join(CACHE_DIR, f'{fn}.json'), 'w') as f:
    f.write(data)
    print("Results written to %s" % os.path.join(CACHE_DIR, f'{fn}.json'))