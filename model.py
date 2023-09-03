import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, Embedding, GlobalAveragePooling1D

class FourierTransformLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Fourier Transform
        return tf.signal.fft(tf.cast(inputs, tf.complex64))

class PointwiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(PointwiseFeedForwardNetwork, self).__init__()

        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')  # (batch_size, seq_len, dff)
        self.dense2 = tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)

    def call(self, x):
        return self.dense2(self.dense1(x))

class FNetBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout=0.1):
        super(FNetBlock, self).__init__()
        
        self.ffn = PointwiseFeedForwardNetwork(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.projection = tf.keras.layers.Dense(d_model)

    def call(self, x, training):
        # Fourier Transform
        fft_out = tf.signal.fft(tf.cast(x, tf.complex64))
        fft_real = tf.math.real(fft_out)
        fft_imag = tf.math.imag(fft_out)
        
        # Concatenate the real and imaginary parts
        fft_concat = tf.concat([fft_real, fft_imag], axis=-1)
        
        fft_concat = self.dropout1(fft_concat, training=training)
        fft_concat = self.projection(fft_concat)

        out1 = self.layernorm1(x + fft_concat)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2