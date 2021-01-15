import tensorflow as tf

class Loss:
    def __init__(self):
        pass

    def cycle_consistency_loss(self, imgs, cycled_imgs, lambda_value):
        loss = lambda_value * tf.reduce_mean(tf.abs(cycled_imgs - imgs))
        return loss

    def gradient_difference_loss(self, preds, gts, gdl_weight=100.0):
        preds_dy, preds_dx = tf.image.image_gradients(preds)
        gts_dy, gts_dx = tf.image.image_gradients(gts)

        gdl_loss = gdl_weight * tf.reduce_mean(tf.abs(tf.abs(preds_dy) - tf.abs(gts_dy)) +
                                                    tf.abs(tf.abs(preds_dx) - tf.abs(gts_dx)))

        return gdl_loss

    def perceptual_loss_fn(self, preds, gts):
        # preds_features = self.vggModel(preds, mode=self.flags.perceptual_mode)
        # gts_features = self.vggModel(gts, mode=self.flags.perceptual_mode)
        #
        # # There are several feature layers
        # perceptual_loss = 0
        # for preds_feature, gts_feature in zip(preds_features, gts_features):
        #     # print('perceputal mode: {}'.format(self.flags.perceptual_mode))
        #     perceptual_loss += tf.reduce_mean(tf.abs(preds_feature - gts_feature))
        #
        # perceptual_loss = self.perceptual_weight * perceptual_loss / len(preds_features)

        # return perceptual_loss
        return 0

    def ssim_loss_fn(self, preds, gts, ssim_weight=0.05):
        # inputs of the ssim should be non-negative
        preds = (preds + 1.) / 2.
        gts = (gts + 1.) / 2.
        ssim_positive = tf.math.maximum(0., tf.image.ssim(preds, gts, max_val=1.0))
        ssim_loss = -tf.math.log(ssim_positive)
        ssim_loss = ssim_weight * tf.reduce_mean(ssim_loss)

        return ssim_loss

    def generator_loss(self, disc_fake):
        d_logit_fake = disc_fake
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))

        return loss

    def discriminator_loss(self, disc_real, disc_fake):
        d_logit_real = disc_real
        d_logit_fake = disc_fake
        # scalar
        error_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
        # scalar
        error_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))

        loss = 0.5 * (error_real + error_fake)
        return loss