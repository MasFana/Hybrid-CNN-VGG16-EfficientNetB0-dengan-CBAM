import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import VGG16, EfficientNetB0
from tensorflow.keras.utils import register_keras_serializable
from PIL import Image, ImageOps


@register_keras_serializable(package="Custom", name="MCDropout")
class MCDropout(layers.Dropout):
    def __init__(self, rate, **kwargs):
        super(MCDropout, self).__init__(rate, **kwargs)

    def call(self, inputs):
        return super().call(inputs, training=True)

    def get_config(self):
        return super().get_config()


def channel_attention_module(x, ratio=8):
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    channel = x.shape[channel_axis]

    shared_layer_one = layers.Dense(
        channel // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )
    shared_layer_two = layers.Dense(
        channel,
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )

    avg_pool = layers.GlobalAveragePooling2D()(x)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_two(shared_layer_one(avg_pool))

    max_pool = layers.GlobalMaxPooling2D()(x)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_two(shared_layer_one(max_pool))

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation("sigmoid")(cbam_feature)
    return layers.Multiply()([x, cbam_feature])


def spatial_attention_module(x):
    def get_pooling_shape(input_shape):
        return input_shape[:-1] + (1,)

    avg_pool = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
        output_shape=get_pooling_shape,
        name="spatial_avg_pool",
    )(x)

    max_pool = layers.Lambda(
        lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
        output_shape=get_pooling_shape,
        name="spatial_max_pool",
    )(x)

    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = layers.Conv2D(
        filters=1,
        kernel_size=7,
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(concat)
    return layers.Multiply()([x, cbam_feature])


def cbam_block(x):
    x = channel_attention_module(x)
    x = spatial_attention_module(x)
    return x


def build_vgg_effattnnet_model():
    input_tensor = Input(shape=(224, 224, 3), name="input_image")

    vgg_base = VGG16(include_top=False, weights=None, input_tensor=input_tensor)
    for layer in vgg_base.layers:
        layer._name = "vgg_" + layer.name

    vgg_out = vgg_base.output
    vgg_flat = layers.Flatten(name="vgg_flatten")(vgg_out)
    vgg_dense = layers.Dense(256, activation="relu", name="vgg_dense")(vgg_flat)
    vgg_branch = MCDropout(0.3, name="vgg_mcd")(vgg_dense)

    eff_base = EfficientNetB0(include_top=False, weights=None, input_tensor=input_tensor)
    for layer in eff_base.layers:
        layer._name = "eff_" + layer.name

    eff_features = eff_base.output
    eff_attn_features = cbam_block(eff_features)
    eff_refined = layers.Add(name="eff_skip_connection")([eff_features, eff_attn_features])
    eff_gap = layers.GlobalAveragePooling2D(name="eff_gap")(eff_refined)
    eff_branch = MCDropout(0.3, name="eff_mcd")(eff_gap)

    merged = layers.Concatenate(name="feature_fusion")([vgg_branch, eff_branch])
    x = layers.Dense(256, activation="relu", name="fusion_dense")(merged)
    x = MCDropout(0.3, name="fusion_mcd")(x)
    output = layers.Dense(6, activation="softmax", name="classification_output")(x)

    model = models.Model(inputs=input_tensor, outputs=output, name="VGG-EffAttnNet_Corrected")
    return model


def load_tf_model_weights(path):
    if not os.path.exists(path):
        return None, f"File '{path}' tidak ditemukan."
    try:
        model = build_vgg_effattnnet_model()
        model.load_weights(path)
        return model, "Success"
    except Exception as e:
        return None, str(e)


def preprocess_tf(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
