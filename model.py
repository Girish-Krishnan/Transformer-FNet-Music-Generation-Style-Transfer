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
    

from tensorflow.keras import layers

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.depth = d_model // self.num_heads
        
        self.Wq = layers.Dense(d_model)
        self.Wk = layers.Dense(d_model)
        self.Wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        d_k = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointwiseFeedForwardNetwork(d_model, dff)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        
    def call(self, x, training):
        attn_output = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
