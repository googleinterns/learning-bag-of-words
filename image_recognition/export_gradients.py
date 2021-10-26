import tensorflow as tf
import os
import random
import pickle as pkl
from glob import glob
from tensorflow.keras import backend as k
from tqdm import tqdm
import argparse
import sklearn.metrics
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

gpus= tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.run_functions_eagerly(True)

parser = argparse.ArgumentParser(description='.')
parser.add_argument('--model', type=str, default='ResNet50', help='Name of the model. E.g. MobileNetV2, ResNet50')
parser.add_argument('--checkpoint', type=str, default='imagenet', help='Checkpoint of the model. Can be a checkpoint name for tf.keras.applications models (e.g. imagenet) or a path')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--num_batches', type=int, default=100, help='Number of batches to be evaluated.')
parser.add_argument('--output_dir', type=str, default='./outputs', help="Root dir.")
parser.add_argument('--imagenet_dir', type=str, default="/scratch/trungvd/imagenet/ILSVRC2012_val")
args = parser.parse_args()

model = getattr(tf.keras.applications, args.model)(weights=args.checkpoint)

model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=1),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

IMAGE_SHAPE = (224, 224)
train_ds = image_dataset_from_directory(
    args.imagenet_dir,
    shuffle=True, seed=2,
    batch_size=args.batch_size,
    class_names=["%03d" % i for i in range(1000)],
    image_size=(224, 224))

train_ds = train_ds.map(lambda x, y: (tf.keras.applications.resnet.preprocess_input(x), y))
print("Dataset loaded")

num_classes = 1000

CACHE_DIR = os.path.join(
    args.output_dir, 
    "%s_%s_bs_%d" % (args.model, 'random' if args.checkpoint is None else args.checkpoint, args.batch_size))
os.makedirs(CACHE_DIR, exist_ok=True)

for i in tqdm(range(args.num_batches), desc="Exporting gradients"):
    grad_path = os.path.join(CACHE_DIR, "grad-%d.npy" % (i))
    label_path = os.path.join(CACHE_DIR, "label-%d.npy" % (i))
    
    image_batch, label_batch = next(iter(train_ds))
    with tf.GradientTape() as tape:
        preds = model(image_batch)
        loss = model.loss(preds, tf.one_hot(label_batch, num_classes))
        grads = tape.gradient(loss, model.get_layer("predictions").trainable_variables[0])
    # print("Batch no. %d" % i)
    # print("Loss:", loss.numpy())
    # print("Grad Norm:", tf.linalg.norm(grads))
    
    # print("Ground-truth Labels:", label_batch.numpy().flatten().tolist())
    # preds = model.predict(image_batch)
    # print("Predicted Labels:", tf.argmax(preds, -1).numpy().flatten().tolist())
    np.save(grad_path, grads.numpy())
    np.save(label_path, label_batch.numpy())