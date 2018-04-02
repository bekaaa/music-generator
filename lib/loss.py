#!/usr/bin/env python2
#-----------------------------------------------------
# this file is taken from the main library,
# refrence : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/loss.py
# I changed some lines to make it work with the project.
# Loss and cost are computed inside this file
# below is the author's Copyright.
#--------------------------------------------------------
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Seq2seq loss operations for use in sequence models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from tensorflow.python.framework import ops
#from tensorflow.python.ops import tf
#from tensorflow.python.ops import tf
#from tensorflow.python.ops import nn_ops
import tensorflow as tf

__all__ = ["sequence_loss"]


def sequence_loss(logits,
				targets,
				weights,
				average_across_timesteps=True,
				average_across_batch=True,
				name=None):
	"""Weighted cross-entropy loss for a sequence of logits.

	Depending on the values of `average_across_timesteps` and
	`average_across_batch`, the return Tensor will have rank 0, 1, or 2 as these
	arguments reduce the cross-entropy at each target, which has shape
	`[batch_size, sequence_length]`, over their respective dimensions. For
	example, if `average_across_timesteps` is `True` and `average_across_batch`
	is `False`, then the return Tensor will have shape `[batch_size]`.

	Args:
	logits: A Tensor of shape
		`[batch_size, sequence_length, num_decoder_symbols]` and dtype float.
		The logits correspond to the prediction across all classes at each
		timestep.
	targets: A Tensor of shape `[batch_size, sequence_length]` and dtype
		int. The target represents the true class at each timestep.
	weights: A Tensor of shape `[batch_size, sequence_length]` and dtype
		float. `weights` constitutes the weighting of each prediction in the
		sequence. When using `weights` as masking, set all valid timesteps to 1
		and all padded timesteps to 0, e.g. a mask returned by `tf.sequence_mask`.
	average_across_timesteps: If set, sum the cost across the sequence
		dimension and divide the cost by the total label weight across timesteps.
	average_across_batch: If set, sum the cost across the batch dimension and
		divide the returned cost by the batch size.
	softmax_loss_function: Function (labels, logits) -> loss-batch
		to be used instead of the standard softmax (the default if this is None).
		**Note that to avoid confusion, it is required for the function to accept
		named arguments.**
	name: Optional name for this operation, defaults to "sequence_loss".

	Returns:
	A float Tensor of rank 0, 1, or 2 depending on the
	`average_across_timesteps` and `average_across_batch` arguments. By default,
	it has rank 0 (scalar) and is the weighted average cross-entropy
	(log-perplexity) per symbol.

	Raises:
	ValueError: logits does not have 3 dimensions or targets does not have 2
				dimensions or weights does not have 2 dimensions.
	"""
	if len(logits.get_shape()) != 2:
		raise ValueError("Logits must be a "
			"[batch_size x sequence_length x logits] tensor")
	if len(targets.get_shape()) != 3:
		raise ValueError("Targets must be a "
			"[batch_size x sequence_length x targits] tensor")
	if len(weights.get_shape()) != 2:
		raise ValueError("Weights must be a [batch_size x sequence_length] "
			"tensor")

	with tf.name_scope(name, "sequence_loss", [logits, targets, weights]):
		note_elements = tf.shape(logits)[-1]
		logits_flat = tf.reshape(logits, [-1, note_elements])
		targets = tf.reshape(targets, [-1, note_elements])
		#---------------------------
		# computing cross entropy
		crossent = 0
		for i in [0, 19, 38] :
			loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets[:,i:i+19],
				logits=logits_flat[:, i:i+19])
			crossent += tf.reduce_mean(loss, axis=1)
		crossent /= 3.
		print(tf.shape(crossent), tf.shape(targets))
		assert tf.shape(crossent) == tf.shape(targets)[0]
		# add weights
		crossent *= tf.reshape(weights, [-1])
		#----------------------------------

		if average_across_timesteps and average_across_batch:
			crossent = tf.reduce_sum(crossent)
			total_size = tf.reduce_sum(weights)
			total_size += 1e-12  # to avoid division by 0 for all-0 weights
			crossent /= total_size
		else:
			batch_size = tf.shape(logits)[0]
			sequence_length = tf.shape(logits)[1]
			crossent = tf.reshape(crossent, [batch_size, sequence_length])
		if average_across_timesteps and not average_across_batch:
			crossent = tf.reduce_sum(crossent, axis=[1])
			total_size = tf.reduce_sum(weights, axis=[1])
			total_size += 1e-12  # to avoid division by 0 for all-0 weights
			crossent /= total_size
		if not average_across_timesteps and average_across_batch:
			crossent = tf.reduce_sum(crossent, axis=[0])
			total_size = tf.reduce_sum(weights, axis=[0])
			total_size += 1e-12  # to avoid division by 0 for all-0 weights
			crossent /= total_size
		return crossent

#if __name__ == '__main__':
#	print('this script should be imported Not run. ')
