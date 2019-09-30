"""TFNetEstimator implementation"""
import enum
import tensorflow as tf
#import tensorflow.estimator as tfe

from . import ops

class SummaryType(enum.IntEnum):
    """Type of value logged to tensorboard"""
    NONE = 0
    VARIABLES = 1
    IMAGES = 2
    AUDIO = 3


tfe = tf.estimator #pylint: disable=invalid-name
#_summary_type_map = {
#    SummaryType.VARIABLES: tfgan_summaries.add_gan_model_summaries,
#    SummaryType.IMAGES: tfgan_summaries.add_gan_model_image_summaries,
#    SummaryType.IMAGE_COMPARISON: tfgan_summaries.add_image_comparison_summaries,
#}

class TFNetEstimator(tfe.Estimator):
    """TFNet Estimator implementation

    model_dir: where checkpoints and model is saved
    time_fn, freq_fn: function that created the graph for time and freq branches
                      at least 1 needs to be defined,
                      these functions takes 2 arguments, input and is_training
    fusion_fn: if both time_fn and freq_fn, this needs to be defined,
               this function should have the signature (time_net, freq_net, is_training)

    TODO:
        document better, some examples?
    """
    def __init__(self, *, #disable positional arguments
                 model_dir=None,
                 time_fn=None,
                 freq_fn=None,
                 fusion_fn=None,
                 loss_fn=None,
                 weight_decay=-1,
                 optimizer=None,
                 get_hooks_fn=None,
                 get_eval_metrics_ops_fn=None,
                 add_summaries=None,
                 config=None,
                ):

        #Basic sanity check
        if not callable(time_fn) and time_fn is not None:
            raise ValueError('time_fn must be callable or None')
        if not callable(freq_fn) and freq_fn is not None:
            raise ValueError('freq_fn must be callable or None')
        if not callable(fusion_fn):
            raise ValueError('fusion_fn must be callable')
        if time_fn is None and freq_fn is None:
            raise ValueError('time_fn and freq_fn cannot be both None')

        def _model_fn(features, labels, mode):
            """TFNet model function
            features, labels are named as such due to tfe.Estimators's
            implementation"""

            tf.logging.info('-------------------------------------------')
            tf.logging.info("Input shapes (features):" + str(features.shape))
            if mode is not tfe.ModeKeys.PREDICT:
                tf.logging.info("Input shapes (labels):" + str(labels.shape))
            tf.logging.info('-------------------------------------------')

            lq_audio = features #rename for better readability in code
            hq_audio = labels #rename for better readability in code

            tfnet_model = _get_tfnet_model(
                mode,
                time_fn, freq_fn, fusion_fn,
                lq_audio,
                add_summaries,
            )

            return _get_estimator_spec(
                mode,
                tfnet_model,
                hq_audio,
                loss_fn,
                optimizer,
                get_hooks_fn,
                get_eval_metrics_ops_fn,
                add_summaries,
                model_dir,
                weight_decay=weight_decay,
                )

        super(TFNetEstimator, self).__init__(model_dir=model_dir,
                                             model_fn=_model_fn,
                                             config=config
                                            )
def _get_tfnet_model(mode,
                     time_fn, freq_fn, fusion_fn,
                     lq_audio,
                     add_summaries,
                    ):
    if mode == tfe.ModeKeys.PREDICT:
        model = _make_prediction_model(time_fn, freq_fn, fusion_fn,
                                       lq_audio, add_summaries)
    elif mode == tfe.ModeKeys.TRAIN:
        model = _make_model(time_fn, freq_fn, fusion_fn,
                            lq_audio,
                            add_summaries, mode)
    elif mode == tfe.ModeKeys.EVAL:
        model = _make_model(time_fn, freq_fn, fusion_fn,
                            lq_audio,
                            add_summaries, mode)

    return model

def _get_estimator_spec(mode,
                        tfnet_model,
                        hq_audio,
                        loss_fn,
                        optimizer,
                        get_hooks_fn,
                        get_eval_metrics_ops_fn,
                        add_summaries,
                        model_dir=None,
                        weight_decay=-1,
                        ):
    is_training = mode == tfe.ModeKeys.TRAIN
    if mode == tfe.ModeKeys.PREDICT:
        estimator_spec = tfe.EstimatorSpec(mode=mode,
                                           predictions=tfnet_model)
    else:
        #this should be extracted
        losses = {'snr': ops.snr_loss(hq_audio, tfnet_model),
                  'l2': ops.l2_loss(hq_audio, tfnet_model),
                  'lsd': ops.lsd_loss(hq_audio, tfnet_model),
                 }

        print('------------------------------------------------')
        if callable(loss_fn):
            print("Using custom loss:", loss_fn)
            loss = loss_fn(hq_audio, tfnet_model)
        else:
            print("Using predefined losses:", loss_fn)
            loss = losses[loss_fn]
            if loss_fn == 'snr':
                loss = -loss

            if weight_decay > 0:
                weights = tf.trainable_variables()
                l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in weights
                                   if 'bias' not in v.name])
                loss = loss + tf.constant(weight_decay) * l2_reg

        if is_training:
            optimizer = optimizer() if callable(optimizer) else optimizer
            if optimizer is None:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            eval_metric_ops = None
        else:
            train_op = None
            eval_metric_ops = {k: tf.metrics.mean(v) for (k, v) in losses.items()}

        _ = [tf.summary.scalar(k, v) for k, v in losses.items()]

        eval_summary_hooks = []
        if add_summaries:
            for _a in add_summaries:
                _a('gt_sample', None, hq_audio)

            eval_summary_hooks.append(tf.train.SummarySaverHook(
                save_steps=100,
                output_dir=model_dir+'/eval',
                summary_op=tf.summary.merge_all('audio_samples')))

        estimator_spec = tfe.EstimatorSpec(mode=mode,
                                           loss=loss,
                                           train_op=train_op,
                                           eval_metric_ops=eval_metric_ops,
                                           evaluation_hooks=eval_summary_hooks
                                          )

    return estimator_spec


def _make_prediction_model(time_fn, freq_fn, fusion_fn, lq_audio,
                           add_summaries):

    time_branch = time_fn(lq_audio, False)
    freq_branch = freq_fn(lq_audio, False)
    model = fusion_fn(time_branch, freq_branch, False)

    if add_summaries:
        for _a in add_summaries:
            _a('pred_sample', model, None)
            _a('input_sample', lq_audio, None)

    return model

def _make_model(time_fn, freq_fn, fusion_fn,
                lq_audio,
                add_summaries, mode):
    is_training = mode == tfe.ModeKeys.TRAIN

    time_branch = time_fn(lq_audio, is_training)
    freq_branch = freq_fn(lq_audio, is_training)
    model = fusion_fn(time_branch, freq_branch, is_training)

    if add_summaries:
        for _a in add_summaries:
            _a('pred_sample', model, None)
            _a('input_sample', lq_audio, None)

    return model
