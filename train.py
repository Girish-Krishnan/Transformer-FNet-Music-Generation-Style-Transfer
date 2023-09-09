from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, Embedding, GlobalAveragePooling1D
from model import *
import pickle
import sys
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import datetime

if len(sys.argv) != 2:
    print("Usage: python train.py <model_type>, where <model_type> is either 'transformer' or 'fnet'")
    exit(1)

model_type = sys.argv[1]
if model_type not in ["transformer", "fnet"]:
    print("Usage: python train.py <model_type>, where <model_type> is either 'transformer' or 'fnet'")
    exit(1)

log_dir = f"logs/{model_type}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

network_input = pickle.load(open("network_input.p", "rb"))
network_output = pickle.load(open("network_output.p", "rb"))
network_artist_input = pickle.load(open("network_artist_input.p", "rb"))
vocabulary = pickle.load(open("vocabulary.p", "rb"))

def build_fnet_model(input_shape, num_artists, num_heads=8, d_model=128, dff=512, num_blocks=4, dropout=0.1):
    inputs = Input(shape=input_shape)
    artist_inputs = Input(shape=(1,))

    # Positional Encoding can be added here if desired
    x = Dense(d_model)(inputs)
    
    # FNet Blocks
    for _ in range(num_blocks):
        x = FNetBlock(d_model, dff, dropout)(x)

    # Average pooling along the sequence length
    x = GlobalAveragePooling1D()(x)
    
    # Artist embedding
    artist_embedding = Embedding(num_artists, d_model)(artist_inputs)
    artist_embedding = tf.squeeze(artist_embedding, axis=1)

    x = tf.add(x, artist_embedding)
    x = Dense(d_model, activation='relu')(x)
    outputs = Dense(input_shape[-1], activation='softmax')(x)
    model = Model(inputs=[inputs, artist_inputs], outputs=outputs)
    return model

def build_transformer_model(input_shape, num_artists, num_heads=8, d_model=128, dff=512, num_blocks=4, dropout=0.1):
    inputs = layers.Input(shape=input_shape)
    artist_inputs = layers.Input(shape=(1,))

    # Positional Encoding can be added here if desired
    x = layers.Dense(d_model)(inputs)
    
    # Transformer Blocks
    for _ in range(num_blocks):
        x = TransformerBlock(d_model, num_heads, dff, dropout)(x)

    # Average pooling along the sequence length
    x = layers.GlobalAveragePooling1D()(x)
    
    # Artist embedding
    artist_embedding = layers.Embedding(num_artists, d_model)(artist_inputs)
    artist_embedding = tf.squeeze(artist_embedding, axis=1)

    x = tf.add(x, artist_embedding)
    x = layers.Dense(d_model, activation='relu')(x)
    outputs = layers.Dense(input_shape[-1], activation='softmax')(x)
    model = Model(inputs=[inputs, artist_inputs], outputs=outputs)
    
    return model

if model_type == "fnet":
    model = build_fnet_model(network_input.shape[1:], len(vocabulary),num_blocks=4)
else:
    model = build_transformer_model(network_input.shape[1:], len(vocabulary),num_blocks=4)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file=f'model_plot_{model_type}.png', show_shapes=True, show_layer_names=True)

# Using a callback to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
checkpoint = ModelCheckpoint(model_type + "_model_epoch_{epoch:02d}.keras", 
                             monitor='val_loss', 
                             verbose=0, 
                             save_best_only=False, 
                             save_weights_only=False, 
                             mode='auto', 
                             save_freq='epoch')

history = model.fit(
    [network_input, network_artist_input],
    network_output,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop, tensorboard_callback, checkpoint]
)

# Save the model
model.save("model.h5")

# Save the history
pickle.dump(history.history, open("history.p", "wb"))


