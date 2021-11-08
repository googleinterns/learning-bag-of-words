# Lint as: python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Speech model."""

import lingvo.compat as tf
from lingvo.core import base_model, optimizer
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.tasks.asr import decoder
from lingvo.tasks.asr import decoder_metrics
from lingvo.tasks.asr import encoder
from lingvo.tasks.asr import frontend as asr_frontend
from lingvo.tools import audio_lib
from lingvo.core import summary_utils
import os
import numpy as np
import json


DecoderTopK = decoder_metrics.DecoderTopK


class AsrModel(base_model.BaseTask):
  """Speech model."""

  @classmethod
  def Params(cls):
    p = super().Params()
    p.encoder = encoder.AsrEncoder.Params()
    p.decoder = decoder.AsrDecoder.Params()
    p.Define(
        'frontend', None,
        'ASR frontend to extract features from input. Defaults to no frontend '
        'which means that features are taken directly from the input.')

    p.Define('decoder_metrics', decoder_metrics.DecoderMetrics.Params(),
             'The decoder metrics layer.')
    # TODO(rpang): switch to use decoder_metrics.include_auxiliary_metrics.
    p.Define(
        'include_auxiliary_metrics', True,
        'In addition to simple WER, also computes oracle WER, SACC, TER, etc. '
        'Turning off this option will speed up the decoder job.')

    tp = p.train
    if tf.flags.FLAGS.export_gradient:
      tp.optimizer = optimizer.SGD.Params()
      tp.learning_rate = 0.1
    else:
      tp.learning_rate = tf.flags.FLAGS.learning_rate
    tp.lr_schedule = (
        schedule.ContinuousSchedule.Params().Set(
          start_step=tf.flags.FLAGS.learning_rate_decay_start,
          half_life_steps=tf.flags.FLAGS.learning_rate_decay_steps,
          min=tf.flags.FLAGS.learning_rate_min))

    tp.vn_start_step = 20000
    tp.vn_std = 0.075
    tp.l2_regularizer_weight = 1e-6
    tp.clip_gradient_norm_to_value = 1.0
    tp.grad_norm_to_clip_to_zero = 100.0
    tp.tpu_steps_per_loop = 20

    return p

  def __init__(self, params):
    if not params.name:
      raise ValueError('params.name not set.')

    params.input.uttid = os.environ['LINGVO_RECONSTRUCTION_UTTID']
    params.input.transcript = []

    assert params.input.uttid
    frames = []
    transcripts = []
    for step_uttid in params.input.uttid.split('|'):
      step_frames = []
      step_transcripts = []
      for uttid in step_uttid.split(','):
        step_frames.append(np.load(f'/home/trungvd/lingvo/librispeech/frames/{uttid}.npy'))
        with open('/home/trungvd/lingvo/librispeech/input/test-sorted.txt') as f:
          lines = f.read().strip().split('\n')
          lines = [l.split(' ', 1) for l in lines]
          step_transcripts.append([l[1] for l in lines if l[0] == uttid][0])
          tf.logging.info("Transcript: %s" % params.input.transcript)
      transcripts.append(step_transcripts)
      frames.append(step_frames)

    params.input.transcript = transcripts

    os.makedirs(tf.flags.FLAGS.logdir, exist_ok=True)
    self.output_path = tf.flags.FLAGS.logdir
    self.output_grads_path = os.path.join(self.output_path, 'grads.pkl')

    dp = params.decoder
    dp.empty_attention_context = False
    if tf.flags.FLAGS.export_gradient:
      dp.empty_attention_context = False
      params.input.num_encoder_steps = 1000
      # params.input.num_encoder_steps = max([max([np.shape(fr)[1] for fr in step_frames]) for step_frames in frames])
    else:
      params.input.num_encoder_steps = 100

    if tf.flags.FLAGS.target_unit == 'word':
      params.input.vocab_size = 16328
      params.input.num_decoder_steps = 15
      # During reconstruction, the length should be specified
      if os.path.exists(os.path.join(tf.flags.FLAGS.logdir, 'info.json')):
        with open(os.path.join(tf.flags.FLAGS.logdir, 'info.json')) as f:
          data = json.load(f)
          params.input.num_decoder_steps = max(max(ls) if type(ls) == list else ls for ls in data['org_tgt_length']) + 1
    elif tf.flags.FLAGS.target_unit == 'char':
      params.input.vocab_size = 76
      params.input.num_decoder_steps = len(params.input.transcript[0]) + 1
    # params.input.num_decoder_steps = 40
    params.decoder.target_seq_len = params.input.num_decoder_steps
    params.encoder.num_frames = params.input.num_encoder_steps
    params.encoder.input_shape = [None, None, params.input.frame_size, 1]

    super().__init__(params)

    p = self.params

    # Construct the model.
    if p.encoder:
      if not p.encoder.name:
        p.encoder.name = 'enc'
      self.CreateChild('encoder', p.encoder)
    if p.decoder:
      if not p.decoder.name:
        p.decoder.name = 'dec'
      self.CreateChild('decoder', p.decoder)
    if p.frontend:
      self.CreateChild('frontend', p.frontend)
    self.CreateChild('decoder_metrics', self._DecoderMetricsParams())

  def _DecoderMetricsParams(self):
    p = self.params
    decoder_metrics_p = p.decoder_metrics.Copy()
    decoder_metrics_p.include_auxiliary_metrics = p.include_auxiliary_metrics
    return decoder_metrics_p

  def _GetDecoderTargets(self, input_batch):
    """Returns targets which will be forwarded to the decoder.

     Subclasses can override this method to change the target that is used by
     the decoder. For example, a subclass could add additional targets that
     can be forwared to the decoder.

    Args:
      input_batch: a NestedMap which contains the targets.

    Returns:
      a NestedMap corresponding to the target selected.
    """
    return input_batch.tgt

  def _MakeDecoderTheta(self, theta, input_batch):
    """Compute theta to be used by the decoder for computing metrics and loss.

    This method can be over-ridden by child classes to add values to theta that
    is passed to the decoder.

    For example, to pass the one hot vector which indicates which data source
    was selected a child class could over-ride this method as follows:

    def _MakeDecoderTheta(self, theta):
      decoder_theta = super(MyModel, self)._MakeDecoderTheta(theta, input_batch)
      decoder_theta.child_onehot = input_batch.source_selected
      return decoder_theta

    Args:
      theta: A `.NestedMap` object containing variable values used to compute
        loss and metrics.
      input_batch: NestedMap containing input data in the current batch. Unused
      here.

    Returns:
      A copy of the decoder theta.
    """
    del input_batch  # Unused
    return theta.decoder.DeepCopy()

  def ComputePredictions(self, theta, input_batch):
    self.input_batch = input_batch
    p = self.params

    if input_batch.src.src_inputs is not None or tf.flags.FLAGS.reconstructed_input == "x":
      encoder_outputs = self._FrontendAndEncoderFProp(theta, input_batch.src)
    else:
      encoder_outputs = None

    tgt = self._GetDecoderTargets(input_batch)

    decoder_theta = self._MakeDecoderTheta(theta, input_batch)
    p = self.params

    predictions = self.decoder.ComputePredictions(decoder_theta, encoder_outputs, tgt, input_batch.src.atten_context)
    predictions.encoder_outputs = None

    self.atten_context = self.decoder.atten_context
    self.softmax_input = self.decoder.softmax_input

    return predictions

  def ComputeLoss(self, theta, predictions, input_batch):
    tgt = self._GetDecoderTargets(input_batch)
    decoder_theta = self._MakeDecoderTheta(theta, input_batch)
    loss = self.decoder.ComputeLoss(decoder_theta, predictions, tgt)
    self.logits = self.decoder.logits
    return loss

  def _FrontendAndEncoderFProp(self,
                               theta,
                               input_batch_src,
                               initial_state=None):
    """FProps through the frontend and encoder.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      input_batch_src: An input NestedMap as per `BaseAsrFrontend.FProp`.
      initial_state: None or a NestedMap object containing the initial states.

    Returns:
      A NestedMap as from `AsrEncoder.FProp`.
    """
    p = self.params
    if p.frontend:
      input_batch_src = self.frontend.FProp(theta.frontend, input_batch_src)
    if initial_state:
      return self.encoder.FProp(
          theta.encoder, input_batch_src, state0=initial_state)
    else:
      return self.encoder.FProp(theta.encoder, input_batch_src)

  def _GetTopK(self, decoder_outs, tag=''):
    return self.decoder_metrics.GetTopK(
        decoder_outs,
        ids_to_strings_fn=self.input_generator.IdsToStrings,
        tag=tag)

  def _ComputeNormalizedWER(self, hyps, refs):
    return self.decoder_metrics.ComputeNormalizedWER(
        hyps, refs, self.params.decoder.beam_search.num_hyps_per_beam)

  def DecodeWithTheta(self, theta, input_batch):
    """Constructs the inference graph."""
    p = self.params
    with tf.name_scope('decode'), tf.name_scope(p.name):
      with tf.name_scope('encoder'):
        encoder_outputs = self._FrontendAndEncoderFProp(theta, input_batch.src)
      with tf.name_scope('beam_search'):
        decoder_outs = self.decoder.BeamSearchDecodeWithTheta(
            theta.decoder, encoder_outputs)

      if py_utils.use_tpu():
        # Decoder metric computation contains arbitrary execution
        # that may not run on TPU.
        dec_metrics = py_utils.RunOnTpuHost(self._ComputeDecoderMetrics,
                                            decoder_outs, input_batch)
      else:
        dec_metrics = self._ComputeDecoderMetrics(decoder_outs, input_batch)
      return dec_metrics

  def _GetTargetForDecoderMetrics(self, input_batch):
    """Returns targets which will be used to compute decoder metrics.

     Subclasses can override this method to change the target that is used when
     calculating decoder metrics.

    Args:
      input_batch: a NestedMap which contains the targets.

    Returns:
      a NestedMap containing 'ids', 'labels', 'paddings', 'weights'
    """
    return self._GetDecoderTargets(input_batch)

  def _ComputeDecoderMetrics(self, decoder_outs, input_batch):
    batch = input_batch.DeepCopy()
    batch.tgt = self._GetTargetForDecoderMetrics(input_batch)
    return self.decoder_metrics.ComputeMetrics(
        decoder_outs,
        batch,
        ids_to_strings_fn=self.input_generator.IdsToStrings)

  def CreateDecoderMetrics(self):
    return self.decoder_metrics.CreateMetrics()

  def PostProcessDecodeOut(self, dec_out_dict, dec_metrics_dict):
    return self.decoder_metrics.PostProcess(dec_out_dict, dec_metrics_dict)

  def Inference(self):
    """Constructs inference subgraphs.

    Returns:
      dict: A dictionary of the form ``{'subgraph_name': (fetches, feeds)}``.
      Each of fetches and feeds is itself a dictionary which maps a string name
      (which describes the tensor) to a corresponding tensor in the inference
      graph which should be fed/fetched from.
    """
    subgraphs = {}
    with tf.name_scope('inference'):
      subgraphs['default'] = self._InferenceSubgraph_Default()
    return subgraphs

  def _InferenceSubgraph_Default(self):
    """Constructs graph for offline inference.

    Returns:
      (fetches, feeds) where both fetches and feeds are dictionaries. Each
      dictionary consists of keys corresponding to tensor names, and values
      corresponding to a tensor in the graph which should be input/read from.
    """
    p = self.params
    with tf.name_scope('default'):
      # TODO(laurenzo): Once the migration to integrated frontends is complete,
      # this model should be upgraded to use the MelAsrFrontend in its
      # params vs relying on pre-computed feature generation and the inference
      # special casing.
      wav_bytes = tf.placeholder(dtype=tf.string, name='wav')
      frontend = self.frontend if p.frontend else None
      if not frontend:
        # No custom frontend. Instantiate the default.
        frontend_p = asr_frontend.MelAsrFrontend.Params()
        frontend = frontend_p.Instantiate()

      # Decode the wave bytes and use the explicit frontend.
      unused_sample_rate, audio = audio_lib.DecodeWav(wav_bytes)
      audio *= 32768
      # Remove channel dimension, since we have a single channel.
      audio = tf.squeeze(audio, axis=1)
      # Add batch.
      audio = tf.expand_dims(audio, axis=0)
      input_batch_src = py_utils.NestedMap(
          src_inputs=audio, paddings=tf.zeros_like(audio))
      input_batch_src = frontend.FPropDefaultTheta(input_batch_src)

      encoder_outputs = self.encoder.FPropDefaultTheta(input_batch_src)
      decoder_outputs = self.decoder.BeamSearchDecode(encoder_outputs)
      topk = self._GetTopK(decoder_outputs)

      feeds = {'wav': wav_bytes}
      fetches = {
          'hypotheses': topk.decoded,
          'scores': topk.scores,
          'src_frames': input_batch_src.src_inputs,
          'encoder_frames': encoder_outputs.encoded
      }

      return fetches, feeds
