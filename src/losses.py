from config import *
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy


def weighted_cross_entropy(y_true, y_pred):
    
    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        return tf.log(y_pred / (1 - y_pred))
    
    y_pred = convert_to_logits(y_pred)
    beta=LOSS_HYPERPARAMETERS['weighted_cross_entropy_beta']
    loss = tf.nn.weighted_cross_entropy_with_logits(
        logits=y_pred,
        targets=y_true,
        pos_weight=beta
    )
    return tf.reduce_mean(loss)


def balanced_cross_entropy(y_true, y_pred):

    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        return tf.log(y_pred / (1 - y_pred))
    
    y_pred = convert_to_logits(y_pred)
    beta = LOSS_HYPERPARAMETERS['balanced_cross_entropy_beta']
    beta = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(
        logits=y_pred,
        targets=y_true,
        pos_weight=pos_weight
    )
    return tf.reduce_mean(loss * (1 - beta))


def focal_loss(y_true, y_pred):

    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.log1p(tf.exp(- tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b
    
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    logits = tf.log(y_pred / (1 - y_pred))
    alpha = LOSS_HYPERPARAMETERS['focal_loss_alpha']
    gamma = LOSS_HYPERPARAMETERS['focal_loss_gamma']
    loss = focal_loss_with_logits(
        logits=logits, targets=y_true,
        alpha=alpha, gamma=gamma, y_pred=y_pred
    )
    return tf.reduce_mean(loss)


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))
    return 1 - numerator / denominator


def tversky_loss(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
    beta = LOSS_HYPERPARAMETERS['tversky_loss_beta']
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
    return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)


def lovasz_softmax_loss(y_true, y_pred):
    '''Code Courtsey: https://github.com/bermanmaxim/LovaszSoftmax/blob/master/tensorflow/lovasz_losses_tf.py'''

    def lovasz_grad(gt_sorted):
        gts = tf.reduce_sum(gt_sorted)
        intersection = gts - tf.cumsum(gt_sorted)
        union = gts + tf.cumsum(1. - gt_sorted)
        jaccard = 1. - intersection / union
        jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
        return jaccard
    
    def lovasz_hinge_flat(logits, labels):

        def compute_loss():
            labelsf = tf.cast(labels, logits.dtype)
            signs = 2. * labelsf - 1.
            errors = 1. - logits * tf.stop_gradient(signs)
            errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
            gt_sorted = tf.gather(labelsf, perm)
            grad = lovasz_grad(gt_sorted)
            loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
            return loss

        loss = tf.cond(
            tf.equal(tf.shape(logits)[0], 0),
            lambda: tf.reduce_sum(logits) * 0.,
            compute_loss, strict=True, name="loss"
        )
        
        return loss
    
    def lovasz_hinge(logits, labels, per_image=True, ignore=None):
        if per_image:
            def treat_image(log_lab):
                log, lab = log_lab
                log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
                log, lab = flatten_binary_scores(log, lab, ignore)
                return lovasz_hinge_flat(log, lab)
            losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
            loss = tf.reduce_mean(losses)
        else:
            loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
        return loss
    
    return lovasz_hinge(labels=y_true, logits=y_pred)


def cross_entropy_plus_dice_loss(y_true, y_pred):
    dice_loss = tf.reshape(dice_loss(y_true, y_pred), (-1, 1, 1))
    loss = binary_crossentropy(y_true, y_pred) + dice_loss
    return loss