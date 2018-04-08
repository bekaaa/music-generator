#!/usr/bin/env python3
#-----------------------------------------------------
# this file is taken from the main library,
# refrence : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/loss.py
# I changed some lines to make it work with the project.
# Loss and cost are computed inside this file
# ==============================================================================
"""Seq2seq loss operations for use in sequence models.
"""
import tensorflow as tf

__all__ = ["sequence_loss"]


def sequence_loss(logits,
				targets,
				weights = None,
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
			"[num_of_elements_in_batch x element_size] tensor")
	if len(targets.get_shape()) != 2:
		raise ValueError("Targets must be a "
			"[num_of_elements_in_batch x element_size] tensor")
	#if len(weights.get_shape()) != 2:
	#	raise ValueError("Weights must be a [batch_size x sequence_length] "
	#		"tensor")

	with tf.name_scope(name, "sequence_loss", [logits, targets, weights]):
		#---------------------------
		# computing cross entropy
		crossent = tf.zeros(logits.get_shape().as_list()[0])
		for i in [0, 19, 38] :
			loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets[:,i:i+19], logits=logits[:, i:i+19])
			# loss is in the same shape of labels and logits
			crossent += tf.reduce_mean(loss, axis=1)
			# axis 1 will remove second axis.
		crossent /= 3.
		assert crossent.get_shape().as_list()[0] == targets.get_shape().as_list()[0]
		# early exit for now
		# reduce crossent to a single number
		crossent = tf.reduce_mean(crossent)
		return crossent

		# add weights
		# crossent *= weights
		# #----------------------------------
		#
		# if average_across_timesteps and average_across_batch:
		# 	crossent = tf.reduce_sum(crossent)
		# 	total_size = tf.reduce_sum(weights)
		# 	total_size += 1e-12  # to avoid division by 0 for all-0 weights
		# 	crossent /= total_size
		# else:
		# 	batch_size = tf.shape(logits)[0]
		# 	sequence_length = tf.shape(logits)[1]
		# 	crossent = tf.reshape(crossent, [batch_size, sequence_length])
		# if average_across_timesteps and not average_across_batch:
		# 	crossent = tf.reduce_sum(crossent, axis=[1])
		# 	total_size = tf.reduce_sum(weights, axis=[1])
		# 	total_size += 1e-12  # to avoid division by 0 for all-0 weights
		# 	crossent /= total_size
		# if not average_across_timesteps and average_across_batch:
		# 	crossent = tf.reduce_sum(crossent, axis=[0])
		# 	total_size = tf.reduce_sum(weights, axis=[0])
		# 	total_size += 1e-12  # to avoid division by 0 for all-0 weights
		# 	crossent /= total_size
		# return crossent

#if __name__ == '__main__':
#	print('this script should be imported Not run. ')
