import math

import tensorflow as tf
from tensorflow import keras


class DLRMScheduler(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, decay_steps, alpha):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha

    def __call__(self, step):
        with tf.name_scope("DLRMSchedule"):
            def warmup():
                scale = global_step_recomp / warmup_steps
                decayed = scale * self.initial_learning_rate
                return decayed

            def cosine_decay(global_step_recomp):
                global_step_recomp = tf.minimum(global_step_recomp, warmup_steps + decay_steps)
                completed_fraction = (global_step_recomp - warmup_steps) / decay_steps
                cosine_decayed = 0.5 * (1.0 + tf.cos(
                    tf.constant(math.pi) * completed_fraction))
                decayed = (1 - self.alpha) * cosine_decayed + self.alpha
                return tf.multiply(initial_learning_rate, decayed)

            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            warmup_steps = tf.cast(self.warmup_steps, dtype)
            global_step_recomp = tf.cast(step, dtype)
            lr = tf.cond(global_step_recomp <= warmup_steps, lambda: warmup(), lambda: cosine_decay(global_step_recomp))
            return lr


if __name__ == '__main__':
    initial_lr, warmup_steps, decay_steps, alpha = 0.01, 20, 10000, 0.0001
    lr_scheduler = DLRMScheduler(initial_lr, warmup_steps, decay_steps, alpha)
    for step in range(30):
        print(step, lr_scheduler(step))
