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
# ==============================================================================
# pylint: disable=line-too-long
r"""Trainer.

To run locally:

.. code-block:: bash

  $ bazel build -c opt //lingvo:trainer
  $ bazel-bin/lingvo/trainer --logtostderr \
      --model=image.mnist.LeNet5 --mode=sync --logdir=/tmp/lenet5 \
      --run_locally=cpu

To use GPU, add `--config=cuda` to build command and set `--run_locally=gpu`.
"""
# pylint: enable=line-too-long
import os
import re
import sys
import json

import time

import lingvo.compat as tf
from lingvo.core import base_model
from lingvo.core import checkpointer
from lingvo.core import cluster_factory
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.core import summary_utils

from lingvo import base_runner

import pickle as pkl
import numpy as np
from datetime import datetime

class Trainer(base_runner.BaseRunner):
  """Trainer on non-TPU."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._job_name = 'trainer'

    with self._graph.as_default(), tf.container(self._container_id):
      self._CreateTF2SummaryWriter(self._train_dir)
      with self._cluster, tf.device(
          self._cluster.GetPlacer()), self._TF2SummaryContext():
        self._model = self.params.Instantiate()
        self._params = self._model.params
        self._model.ConstructFPropBPropGraph()
      self._CreateTF2SummaryOps()
      self._initialize_tables = tf.tables_initializer()
      self._initialize_local_vars = tf.local_variables_initializer()
      self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
      tf.logging.info('Trainer number of enqueue ops: %d',
                      len(self.enqueue_ops))

    try:
      self._task_probs_summary_writers = []
      for task in self._model.task_schedule.tasks:
        path = os.path.join(os.path.join(self._train_dir, task))
        tf.io.gfile.makedirs(path)
        self._task_probs_summary_writers.append(self._CreateSummaryWriter(path))
    except AttributeError:
      tf.logging.info('AttributeError. Expected for single task models.')
      self._task_probs_summary_writers = []

    self._step_rate_tracker = summary_utils.StepRateTracker()

    # Saves the graph def.
    if self.params.cluster.task > 0:
      self._summary_writer = None
    else:
      self._WriteToLog(self.params.ToText(), self._train_dir,
                       'trainer_params.txt')
      self._summary_writer = self._CreateSummaryWriter(self._train_dir)
      tf.io.write_graph(self._graph.as_graph_def(), self._train_dir,
                        'train.pbtxt')
    worker_id = self.params.cluster.task
    self._start_up_delay_steps = (((worker_id + 1) * worker_id / 2) *
                                  self.params.train.start_up_delay_steps)

  def _SummarizeValue(self, steps, tag, value, writer=None):
    self._summary_writer.add_summary(
        tf.Summary(
          value=[tf.Summary.Value(tag=tag, simple_value=value)]), steps)

  def Start(self):
    self._RunLoop('trainer', self._Loop)

  def StartEnqueueOp(self, op):
    self._RunLoop(
        'trainer/enqueue_op/%s' % op.name, self._LoopEnqueue, loop_args=[op])

  def _LoopEnqueue(self, op):
    # Evaler/Controller jobs may find that the trial is infeasible and report
    # done earlier. This is an important check since the trainer may retry
    # indefinitely without it.
    if self._trial.ShouldStop():
      tf.logging.info('Training skipped (trial requested to stop).')
      return
    return super()._LoopEnqueue(op)

  def _Loop(self):
    # tf.random.set_seed(0)
    # np.random.seed(0)
    # Evaler/Controller jobs may find that the trial is infeasible and report
    # done earlier. This is an important check since the trainer may retry
    # indefinitely without it.
    if self._trial.ShouldStop():
      tf.logging.info('Training skipped (trial requested to stop).')
      return

    with tf.container(self._container_id), self._cluster, self._GetSession() as sess:
      self._train_dir = os.path.join('/home/trungvd/lingvo/models')
      saver = tf.train.Saver(
        var_list=[var for var in tf.global_variables() if 'reconstructed' not in var.op.name and var.op.name not in ['client_grad']],
        write_version=tf.train.SaverDef.V2)

      def write_info(d: dict):
        with open(os.path.join(tf.flags.FLAGS.logdir, 'info.json'), 'w') as f:
          tf.logging.info("Info: " + str(d))
          json.dump(d, f)

      def read_info():
        try:
          with open(os.path.join(tf.flags.FLAGS.logdir, 'info.json'), 'r') as f:
            return json.load(f)
        except FileNotFoundError:
          return dict()

      info = read_info()

      saver.restore(sess, tf.flags.FLAGS.checkpoint_path)
      sess.run(tf.assign(self._model.global_step, 0))

      # sess.run(tf.global_variables_initializer())
      sess.run(tf.variables_initializer([v for v in tf.global_variables() if "reconstructed" in v.op.name]))

      # This initializes local tables
      sess.run(self._initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)

      self._InitializeTF2SummaryWriter(sess)
      for task in self._model.tasks:
        task.input.Initialize(sess)
      global_step = self._WaitUntilInit(sess, self._start_up_delay_steps)

      status_interval_steps = 100
      next_status_step = 1
      eval_metrics = None

      def assign_inputs(reconstructed_input=None, one_hot=None):
        if one_hot is not None:
          if type(one_hot) == str:
            if one_hot == 'random':
              sess.run([
                tf.assign(
                  task.input_batch.tgt.rec_one_hot,
                  tf.constant(np.random.randn(*task.input_batch.tgt.rec_one_hot.get_shape()), dtype=tf.float32)),
                # tf.assign(task.input_batch.tgt.paddings, task.input_batch.tgt.org_paddings)
              ])
            elif one_hot == 'random_one_hot':
              _, seq_len, num_classes = task.input_batch.tgt.rec_one_hot.get_shape()
              sess.run([
                tf.assign(
                  task.input_batch.tgt.rec_one_hot,
                  tf.one_hot(tf.constant(np.random.randint(0, num_classes, (1, seq_len)), dtype=tf.int32), depth=num_classes)),
                # tf.assign(task.input_batch.tgt.paddings, task.input_batch.tgt.org_paddings)
              ])
            elif one_hot == 'zero':
              sess.run([
                tf.assign(
                  task.input_batch.tgt.rec_one_hot,
                  tf.constant(np.zeros(task.input_batch.tgt.rec_one_hot.get_shape()), dtype=tf.float32)),
                # tf.assign(task.input_batch.tgt.paddings, task.input_batch.tgt.org_paddings)
              ])
          elif one_hot is not None:
            sess.run([
              tf.assign(task.input_batch.tgt.rec_one_hot, one_hot),
              # tf.assign(task.input_batch.tgt.paddings, task.input_batch.tgt.org_paddings)
            ])

        if type(reconstructed_input) == str:
          if reconstructed_input == 'random':
            sess.run([tf.assign(
              task.input_batch.reconstructed_input,
              tf.constant(np.random.randn(*task.input_batch.reconstructed_input.get_shape()), dtype=tf.float32))])
          else:
            raise ValueError(reconstructed_input + " is not valid.")
        elif reconstructed_input is not None:
          sess.run([tf.assign(task.input_batch.reconstructed_input, reconstructed_input)])

      if tf.flags.FLAGS.export_gradient:
        _weights_prev = sess.run(task.learners[0].weights)
        num_steps = len(task.input_batch.src.org_src_inputs)
        for i in range(num_steps):
          assign_inputs(
            reconstructed_input=task.input_batch.src.org_src_inputs[i],
            one_hot=task.input_batch.tgt.org_one_hot[i])
          _ = sess.run([task.train_op])
        _weights = sess.run(task.learners[0].weights)
        _lr = sess.run(task.learners[0].LearningRate())

        _x, _one_hot, _grad, _losses, _atten_context, _softmax_input = sess.run([
          task.input_batch.src.src_inputs,
          task.input_batch.tgt.one_hot,
          task.learners[0].grad,
          task.learners[0].losses,
          task.atten_context,
          task.softmax_input,
        ])

        with open(task.output_grads_path, 'wb') as f:
          pkl.dump(dict(
            x=_x,
            one_hot=_one_hot,
            grads=(_weights_prev - _weights) / _lr,
            atten_context=_atten_context,
            softmax_input=_softmax_input
          ), f)

        tf.logging.info("Gradients exported to %s" % task.output_grads_path)
      else:
        tf.logging.info("Loading gradients from %s" % task.output_grads_path)
        with open(task.output_grads_path, 'rb') as f:
          data = pkl.load(f)
          _x = data['x']
          _one_hot = data['one_hot']
          _grad = data['grads']
          _atten_context = data['atten_context']
          _softmax_input = data['softmax_input']

        sess.run([tf.assign(task.learners[0].client_grad, tf.constant(_grad))])

        org_input = None
        if tf.flags.FLAGS.reconstructed_input == 'x':
          org_input = task.input_batch.src.org_src_inputs
        elif tf.flags.FLAGS.reconstructed_input == 'atten_context':
          org_input = tf.constant(_atten_context)
        elif tf.flags.FLAGS.reconstructed_input == 'softmax_input':
          org_input = tf.constant(_softmax_input)

        if False and not tf.flags.FLAGS.export_gradient:
          # check gradients
          assign_inputs(
            # reconstructed_input=org_input[:, :task.input_batch.tgt.length[0], :],
            reconstructed_input=org_input,
            one_hot=task.input_batch.tgt.org_one_hot_restricted)

          grad_dist, inp_norm, out_norm = sess.run([
            task.learners[0].grads_dist,
            # tf.norm(task.input_batch.reconstructed_input - org_input[:, :task.input_batch.tgt.length[0], :]),
            tf.norm(task.input_batch.reconstructed_input - org_input),
            tf.norm(task.input_batch.tgt.one_hot - task.input_batch.tgt.org_one_hot)
          ])
          assert grad_dist < 1e-4, "Gradient distance is %.9f" % grad_dist

      info['org_tgt_length'] = [[int(l) - 1 for l in ls] for ls in sess.run(task.input_batch.tgt.org_length)]
      # info['org_src_length'] = int(task.input_batch.src.org_src_inputs.shape[1])
      info['org_tgt_labels'] = [sess.run(ids[:, 1:]).tolist() for ids in task.input_batch.tgt.org_ids]
      info['org_tgt_str_labels'] = [[w.decode() for w in sess.run(lbls).tolist()] for lbls in task.input_batch.tgt.org_str_labels]
      if task.input_batch.tgt.bow is not None:
        info['rec_bow'] = task.input_batch.tgt.bow
        info['rec_tgt_length'] = task.input_batch.tgt.rec_length - len(info['org_tgt_length'])
      if 'results' not in info:
        info['results'] = []

      write_info(info)

      if tf.flags.FLAGS.no_rec or tf.flags.FLAGS.export_gradient:
        exit(0)

      # initialize reconstructed objects
      assign_inputs(
        reconstructed_input='random',
        one_hot='random_one_hot',
      )

      # Initialize inputs and targets
      
      # sess.run([tf.assign(task.input_batch.src.src_inputs, tf.constant(_x + 0.1 * np.random.randn(*_x.shape), dtype=tf.float32))])
      # sess.run([tf.assign(task.input_batch.src.src_inputs, tf.constant(np.random.randn(*_x.shape), dtype=tf.float32))])
      # sess.run([tf.assign(task.input_batch.src.src_inputs, tf.constant(np.zeros(_x.shape), dtype=tf.float32))])

      # Add metrics

      def cosine_similarity(a, b):
        a = tf.nn.l2_normalize(a, 1)
        b = tf.nn.l2_normalize(b, 1)
        return tf.reduce_sum(tf.multiply(a, b))

      metrics = dict(
        grad_dist=task.learners[0].grads_dist,
        # inp_mae=tf.reduce_mean(tf.abs(task.input_batch.reconstructed_input - org_input[:, :tf.shape(task.input_batch.reconstructed_input)[1], :])),
        ymse=tf.reduce_mean(tf.norm(task.input_batch.tgt.one_hot - task.input_batch.tgt.org_one_hot, ord=1, axis=-1)),
        lbl_cosine_sim=cosine_similarity(
          task.input_batch.tgt.one_hot, task.input_batch.tgt.org_one_hot),
        lbl_edit_dist=tf.edit_distance(
          tf.sparse.from_dense(task.input_batch.tgt.rec_labels),
          tf.sparse.from_dense(task.input_batch.tgt.org_ids[0][:, 1:]), normalize=True)
      )

      # if task.input_batch.src.src_inputs is not None:
      #   metrics["xnmae"] = tf.norm(tf.math.l2_normalize(task.input_batch.src.src_inputs) - tf.math.l2_normalize(tf.constant(_x)), 1) / task.params.input.num_encoder_steps
      #   metrics["xmae"] = tf.norm(task.input_batch.src.src_inputs - tf.constant(_x), 1) / task.params.input.num_encoder_steps / task.params.input.frame_size

      prev_rec_transcript = []
      log_file = open(os.path.join(task.output_path, 'log.log'), 'w')
      log_file.write('1.0\n')
      result = {}
      result['start_time'] = str(datetime.now())
      while True:
        if (self._trial.ShouldStopAndMaybeReport(global_step, eval_metrics) or
            self._ShouldStop(sess, global_step)):
          tf.logging.info('Training finished.')
          if self._early_stop:
            time.sleep(300)  # controller hangs if it doesn't finish first
          self._DequeueThreadComplete()
          return

        # If a task is explicitly specified, only train that task.
        if self._model_task_name:
          task = self._model.GetTask(self._model_task_name)
        else:
          # Note: This is a slightly stale global_step value from the previous
          # sess.run() call.
          # For multi-task models, `self._model.task_schedule.cur_probs` will
          # be updated.
          task = self._model.SampleTask(global_step)
          if self._task_probs_summary_writers:
            for index, prob in enumerate(self._model.task_schedule.cur_probs):
              self._SummarizeValue(global_step, 'task_probability', prob,
                                   self._task_probs_summary_writers[index])
            try:
              for index, task in enumerate(self._model.tasks):
                self._SummarizeValue(global_step, 'task_weight',
                                     sess.run(task.vars.task_weight),
                                     self._task_probs_summary_writers[index])
            except AttributeError:
              pass

        (_, eval_metrics, _metrics, per_example_tensors, rec_transcript, org_transcript, weights, _global_step, _one_hot, _lr) = sess.run([
          task.train_op,
          task.eval_metrics,
          metrics,
          task.per_example_tensors,
          task.input_batch.tgt.rec_str_labels,
          task.input_batch.tgt.org_str_labels,
          task.input_batch.tgt.weights,
          self._model.global_step,
          task.input_batch.tgt.rec_one_hot,
          task.learners[0].lr
        ])

        # sess.run([
        #   tf.assign(task.input_batch.tgt.rec_one_hot, 1 - tf.nn.relu(1 - tf.nn.relu(task.input_batch.tgt.rec_one_hot))),
        # ])

        prev_rec_transcript.append(rec_transcript.tolist()[0])
        frac_unchanged_steps = 0.
        if len(prev_rec_transcript) > tf.flags.FLAGS.num_unchanged_steps:
          prev_rec_transcript = prev_rec_transcript[-tf.flags.FLAGS.num_unchanged_steps:]
          most_frequent_transcript = max(set(prev_rec_transcript), key=prev_rec_transcript.count)
          frac_unchanged_steps = prev_rec_transcript.count(most_frequent_transcript) / tf.flags.FLAGS.num_unchanged_steps
          if rec_transcript == most_frequent_transcript and frac_unchanged_steps > 0.99:
            log_file.write('Done')

            result['rec_tgt_labels'] = sess.run(task.input_batch.tgt.rec_labels).tolist(),
            result['rec_tgt_str_labels'] = [w.decode() for w in sess.run(task.input_batch.tgt.rec_str_labels).tolist()],
            result['num_steps'] = int(_global_step)
            # result['rec_src_mae'] = float(_metrics['inp_mae']) if 'inp_mae' in _metrics else None
            result['loss'] = float(_metrics['grad_dist'])
            result['finished_time'] = str(datetime.now())
            info['results'].append(result)
            write_info(info)
            log_file.close()
            exit(0)

        for tag, val in _metrics.items():
          self._SummarizeValue(_global_step, tag, val)

        # text_tensor = tf.make_tensor_proto(task.input_batch.tgt.reconstructed_labels, dtype=tf.string)
        # meta = tf.SummaryMetadata()
        # meta.plugin_data.plugin_name = "text"
        # summary = tf.Summary()
        # summary.value.add(tag="transcript", metadata=meta, tensor=text_tensor)
        # self._summary_writer.add_summary(summary)

        self._summary_writer.flush()

        # Explicitly fetch global_step after running train_op.
        # TODO(b/151181934): Investigate this behavior further.

        if _global_step % 20 == 0:
          print(_one_hot)
          log_str = ', '.join([
            "loss:%.4f" % _metrics['grad_dist'],
            "lbl_edit_dist:%.4f" % _metrics['lbl_edit_dist'],
            "ymse:%4f" % _metrics['ymse'],
            "text:%s" % rec_transcript[0].decode(),
            "org_text:%s" % org_transcript[0][0].decode(),
            "unchanged:%.2f" % frac_unchanged_steps,
            "lr:%.2f" % _lr,
          ])
          tf.logging.info(log_str)
          log_file.write(log_str + '\n')

        # task.ProcessFPropResults(sess, task_global_step, eval_metrics, per_example_tensors)
        # self._RunTF2SummaryOps(sess)

        # self._model.ProcessFPropResults(sess, global_step, eval_metrics, per_example_tensors)


def GetDecoderDir(logdir, decoder_type, model_task_name):
  if model_task_name:
    decoder_dir = '%s_%s' % (decoder_type, model_task_name)
  else:
    decoder_dir = decoder_type
  return os.path.join(logdir, decoder_dir)


def _GetCheckpointIdForDecodeOut(ckpt_id_from_file, global_step):
  """Retrieve the checkpoint id for the decoder out file.

  Compares the checkpoint id found in the checkpoint file name to global
  step. If they diverge, uses the retrieved id and prints a warning.

  Args:
   ckpt_id_from_file: Checkpoint Id from the checkpoint file path.
   global_step: int specifying the global step of the model.

  Returns:
   Checkpoint id as int.
  """
  tf.logging.info('Loaded checkpoint is at global step: %d', global_step)
  tf.logging.info('Checkpoint id according to checkpoint path: %d',
                  ckpt_id_from_file)
  if global_step != ckpt_id_from_file:
    tf.logging.warning(
        'Checkpoint id %d != global step %d. '
        'Will use checkpoint id from checkpoint file for '
        'writing decoder output.', ckpt_id_from_file, global_step)
  return ckpt_id_from_file


class Decoder(base_runner.BaseRunner):
  """Decoder."""

  def __init__(self, decoder_type, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._job_name = 'decoder_' + decoder_type
    self.params.cluster.do_eval = True
    self._cluster = cluster_factory.Cluster(self.params.cluster)
    self._decoder_dir = GetDecoderDir(self._logdir, self._job_name,
                                      self._model_task_name)
    tf.io.gfile.makedirs(self._decoder_dir)

    self._decode_path = None
    # Multitask params doesn't have 'task'.
    if 'task' in self.params:
      self._decode_path = checkpointer.GetSpecificCheckpoint(
          self.params.task.eval.load_checkpoint_from)

    self._summary_writer = self._CreateSummaryWriter(self._decoder_dir)
    self._should_report_metrics = self._job_name.startswith(
        self.params.reporting_job)

    with self._graph.as_default(), tf.container(self._container_id):
      self._CreateTF2SummaryWriter(self._decoder_dir)
      with self._cluster, tf.device(
          self._cluster.GetPlacer()), self._TF2SummaryContext():
        self._model = self.params.Instantiate()
        self._params = self._model.params
        self._task = self._model.GetTask(self._model_task_name)
        # Note, different graphs are being constructed for different model
        # tasks, which may result in different node names being chosen.
        # Obviously, variable names has to be stay the same between train and
        # decode.
        cluster = self._cluster
        with tf.device(cluster.input_device):
          input_batch = (self._task.input_generator.GetPreprocessedInputBatch())

        self._dec_output = self._task.Decode(input_batch)
        self._summary_op = tf.summary.merge_all()
        self.checkpointer = self._CreateCheckpointer(self._train_dir,
                                                     self._model)
      self._CreateTF2SummaryOps()
      self._initialize_tables = tf.tables_initializer()
      self._initialize_local_vars = tf.local_variables_initializer()
      # No queues are allowed for decoder models.
      self.enqueue_ops = tf.get_collection(py_utils.ENQUEUE_OPS)
      assert not self.enqueue_ops

    # Saves the graph def.
    self._WriteToLog(self.params.ToText(), self._decoder_dir, 'params.txt')
    if self.params.cluster.task == 0:
      tf.io.write_graph(self._graph.as_graph_def(), self._decoder_dir,
                        '%s.pbtxt' % self._job_name)

  def _CreateCheckpointer(self, train_dir, model):
    """Wrapper method for override purposes."""
    return checkpointer.Checkpointer(train_dir, model)

  def Start(self):
    self._RunLoop(self._job_name, self._Loop)

  def _Loop(self):
    with tf.container(self._container_id), self._cluster, self._GetSession(
        inline=False) as sess:
      # This initializes local tables
      sess.run(self._initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)
      self._InitializeTF2SummaryWriter(sess)
      self._task.input.Initialize(sess)

      if self._decode_path:
        self.DecodeCheckpoint(sess, self._decode_path)
      else:
        path = None
        while True:
          path = self._FindNewCheckpoint(path, sess)
          if not path or self.DecodeCheckpoint(sess, path):
            break

    # Maybe decode the last checkpoint if we are not given a specific
    # checkpoint to decode.
    if self._decode_path is None:
      self.DecodeLatestCheckpoint(path)

    if self._should_report_metrics:
      tf.logging.info('Reporting trial done.')
      self._trial.ReportDone()
    tf.logging.info('Decoding finished.')

  @classmethod
  def GetDecodeOutPath(cls, decoder_dir, checkpoint_id):
    """Gets the path to decode out file."""
    out_dir = cls._GetTtlDir(decoder_dir, duration='7d')
    return os.path.join(out_dir, 'decoder_out_%09d' % checkpoint_id)

  def GetCkptIdFromFile(self, checkpoint_path):
    return int(re.sub(r'.*ckpt-', '', checkpoint_path))

  def DecodeCheckpoint(self, sess, checkpoint_path):
    """Decodes `samples_per_summary` examples using `checkpoint_path`."""
    p = self._task.params
    ckpt_id_from_file = self.GetCkptIdFromFile(checkpoint_path)
    if ckpt_id_from_file < p.eval.start_decoder_after:
      return False
    samples_per_summary = p.eval.decoder_samples_per_summary
    if samples_per_summary is None:
      samples_per_summary = p.eval.samples_per_summary
    if samples_per_summary == 0:
      assert self._task.params.input.resettable
    self.checkpointer.RestoreFromPath(sess, checkpoint_path)

    global_step = sess.run(py_utils.GetGlobalStep())

    if self._task.params.input.resettable:
      tf.logging.info('Resetting input_generator.')
      self._task.input.Reset(sess)

    dec_metrics = self._task.CreateDecoderMetrics()
    if not dec_metrics:
      tf.logging.info('Empty decoder metrics')
      return
    buffered_decode_out = []
    num_examples_metric = dec_metrics['num_samples_in_batch']
    start_time = time.time()
    while samples_per_summary == 0 or (num_examples_metric.total_value <
                                       samples_per_summary):
      try:
        tf.logging.info('Fetching dec_output.')
        fetch_start = time.time()
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=False)
        if self._summary_op is None:
          # No summaries were collected.
          dec_out = sess.run(self._dec_output, options=run_options)
        else:
          dec_out, summary = sess.run([self._dec_output, self._summary_op],
                                      options=run_options)
          self._summary_writer.add_summary(summary, global_step)
        self._RunTF2SummaryOps(sess)
        post_process_start = time.time()
        tf.logging.info('Done fetching (%f seconds)' %
                        (post_process_start - fetch_start))
        decode_out = self._task.PostProcessDecodeOut(dec_out, dec_metrics)
        if decode_out:
          buffered_decode_out.extend(decode_out)
        tf.logging.info(
            'Total examples done: %d/%d '
            '(%f seconds decode postprocess)', num_examples_metric.total_value,
            samples_per_summary,
            time.time() - post_process_start)
      except tf.errors.OutOfRangeError:
        if not self._task.params.input.resettable:
          raise
        break
    tf.logging.info('Done decoding ckpt: %s', checkpoint_path)

    summaries = {k: v.Summary(k) for k, v in dec_metrics.items()}
    elapsed_secs = time.time() - start_time
    example_rate = num_examples_metric.total_value / elapsed_secs
    summaries['examples/sec'] = metrics.CreateScalarSummary(
        'examples/sec', example_rate)
    summaries['total_samples'] = metrics.CreateScalarSummary(
        'total_samples', num_examples_metric.total_value)
    self._WriteSummaries(
        self._summary_writer,
        os.path.basename(self._decoder_dir),
        global_step,
        summaries,
        text_filename=os.path.join(self._decoder_dir,
                                   'score-{:08d}.txt'.format(global_step)))
    self._ExportMetrics(
        # Metrics expects python int, but global_step is numpy.int64.
        decode_checkpoint=int(global_step),
        dec_metrics=dec_metrics,
        example_rate=example_rate)
    # global_step and the checkpoint id from the checkpoint file might be
    # different. For consistency of checkpoint filename and decoder_out
    # file, use the checkpoint id as derived from the checkpoint filename.
    checkpoint_id = _GetCheckpointIdForDecodeOut(ckpt_id_from_file, global_step)
    decode_out_path = self.GetDecodeOutPath(self._decoder_dir, checkpoint_id)

    decode_finalize_args = base_model.DecodeFinalizeArgs(
        decode_out_path=decode_out_path, decode_out=buffered_decode_out)
    self._task.DecodeFinalize(decode_finalize_args)

    should_stop = global_step >= self.params.train.max_steps
    if self._should_report_metrics:
      tf.logging.info('Reporting eval measure for step %d.' % global_step)
      trial_should_stop = self._trial.ReportEvalMeasure(global_step,
                                                        dec_metrics,
                                                        checkpoint_path)
      should_stop = should_stop or trial_should_stop
    return should_stop

  def DecodeLatestCheckpoint(self, last_path=None):
    """Runs decoder on the latest checkpoint."""
    with tf.container(
        self._container_id), self._cluster, self._GetSession() as sess:
      # This initializes local tables
      sess.run(self._initialize_tables)
      # This initializes local variables.
      sess.run(self._initialize_local_vars)
      self._task.input.Initialize(sess)
      path = tf.train.latest_checkpoint(self._train_dir)
      if not path:
        tf.logging.info('No checkpoint available.')
        return
      elif path == last_path:
        tf.logging.info('Latest checkpoint was already decoded.')
        return
      self.DecodeCheckpoint(sess, path)
