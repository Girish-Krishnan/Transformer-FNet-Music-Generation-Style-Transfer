from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, Embedding, GlobalAveragePooling1D
from model import *
import pickle

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

model = build_fnet_model(network_input.shape[1:], len(vocabulary))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Using a callback to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(
    [network_input, network_artist_input],
    network_output,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Save the model
model.save("model.keras")

# Save the history
pickle.dump(history.history, open("history.p", "wb"))


