This repo stores the official code for the paper

```
@inproceedings{dang2021revealing,
 author = {Dang, Trung and Thakkar, Om and Ramaswamy, Swaroop and Mathews, Rajiv and Chin, Peter and Beaufays, Françoise},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Revealing and Protecting Labels in Distributed Training},
 url = {},
 year = {2021}
}
```

# Instructions

## Reveal image labels

The following scripts export gradients computed by keras image recognition backbones and performs RLG to reveal set of labels in the batch. Results are saved into `results_svd.json`.

```
cd image_recognition
python export_gradients.py --model ResNet50 --batch_size 10 --imagenet_dir <path_to_imagenet>
python rlg.py --model ResNet50 --batch_size 10
```

## Reveal speech transcript

To reveal the transcript(s) from gradients of lingvo's ASR model, follow these steps:

- Clone [lingvo](https://github.com/tensorflow/lingvo) (we use the commit #d5aada292071f17967730ca26d8aea7b8f5a31c2)
- Copy files in `speech_recognition/lingvo` over lingvo's code. These files contain minor changes to lingvo's source code to make the model twice differentiable (dynamic -> static RNN, replace while_loop with for-loop).
- Build lingvo with bazel (see instructions [here](https://github.com/tensorflow/lingvo#installation), also [docker](https://github.com/tensorflow/lingvo/blob/master/docker/dev.dockerfile))
- Run this script to reconstruct a single utterance

```
export LINGVO_RECONSTRUCTION_UTTID=<librispeech_utt_id>
export LINGVO_RECONSTRUCTION_TAG=wpm-rand-bow
export OUTPUT_DIR=/home/trungvd/lingvo-outputs/$LINGVO_RECONSTRUCTION_TAG/$LINGVO_RECONSTRUCTION_UTTID

rm -r $OUTPUT_DIR/train

export REC_PARAMS="--model=asr.librispeech.Librispeech960Wpm \
  --checkpoint_path <path_to_a_Librispeech960Wpm_checkpoint> \
  --model_tag $LINGVO_RECONSTRUCTION_TAG \
  --job trainer \
  --mode=async --logdir=$OUTPUT_DIR \
  --logtostderr \
  --saver_max_to_keep=2 \
  --run_locally=gpu \
  --learning_rate 0.05 \
  --learning_rate_min 0.005 \
  --learning_rate_decay_steps 4000 \
  --num_unchanged_steps 2000 \
  --reconstructed_input atten_context \
  --target_unit word"

echo "Started $(date)" >> $HISTORY_PATH

../bazel-bin/lingvo/trainer $REC_PARAMS --export_gradient
echo "Gradient exported"

echo "Inferring bag of words..."
python ../scripts/infer_bow.py --uttid=$LINGVO_RECONSTRUCTION_UTTID --tag=$LINGVO_RECONSTRUCTION_TAG --target_unit $LINGVO_REC_UNIT

../bazel-bin/lingvo/trainer $REC_PARAMS --use_bow
```

List of utt ids used for our experiments is stored in `speech_recognition/lingvo/lingvo/data`