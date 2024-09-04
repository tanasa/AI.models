
""" an inspiration from : https://www.analyticsvidhya.com/blog/2023/07/generative-ai-with-vaes-gans-transformers/ """

# VAE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the encoder network
encoder_inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(hidden_dim, activation="relu")(encoder_inputs)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# Define the decoder network
decoder_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(hidden_dim, activation="relu")(decoder_inputs)
decoder_outputs = layers.Dense(output_dim, activation="sigmoid")(x)


""" Define Sampling Function
The sampling function takes the mean and log variance of a latent space as inputs and 
generates a random sample by adding noise scaled by the exponential of half the log variance to the mean.
"""

# Define the sampling function for the latent space
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(batch_size, latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])



""" Define Loss Function
The VAE loss function has the reconstruction loss, which measures the similarity between the input and output, 
and the Kullback-Leibler (KL) loss, which regularizes the latent space by penalizing deviations from a prior distribution. 
These losses are combined and added to the VAE model allowing for end-to-end training that simultaneously optimizes both 
the reconstruction and regularization objectives.
"""

vae = keras.Model(inputs=encoder_inputs, outputs=decoder_outputs)

# Define the loss function
reconstruction_loss = keras.losses.binary_crossentropy(encoder_inputs, decoder_outputs)
reconstruction_loss *= input_dim

kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
kl_loss = tf.reduce_mean(kl_loss) * -0.5

vae_loss = reconstruction_loss + kl_loss
vae.add_loss(vae_loss)

"""Compile and train the model"""

# Compile and train the VAE
vae.compile(optimizer="adam")
vae.fit(x_train, epochs=epochs, batch_size=batch_size)



# Generative Adversarial Networks (GANs)

"""
The generator aims to produce realistic samples, while the discriminator distinguishes between real and generated samples.
"""


""" A generator network, represented by the ‘generator’ variable, which takes a latent space input and 
transforms it through a series of dense layers with ReLU activations to generate synthetic data samples.

Similarly, it also defines a discriminator network, represented by the ‘discriminator’ variable, which takes the generated data samples as input and 
passes them through dense layers with ReLU activations to predict a single output value indicating the probability of the input being real or fake.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the generator network
generator = keras.Sequential([
    layers.Dense(256, input_dim=latent_dim, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(output_dim, activation="sigmoid")
])

# Define the discriminator network
discriminator = keras.Sequential([
    layers.Dense(512, input_dim=output_dim, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

""" The GAN model is defined by combining the generator and discriminator networks. 
The discriminator is compiled separately with binary cross-entropy loss and the Adam optimizer. 
During GAN training, the discriminator is frozen to prevent its weights from being updated. 
The GAN model is then compiled with binary cross-entropy loss and the Adam optimizer."""

# Define the GAN model
gan = keras.Sequential([generator, discriminator])

# Compile the discriminator
discriminator.compile(loss="binary_crossentropy", optimizer="adam")

# Freeze the discriminator during GAN training
discriminator.trainable = False

# Compile the GAN
gan.compile(loss="binary_crossentropy", optimizer="adam")

""" Training the GAN
In the training loop, the discriminator and generator are trained separately using batches of real and generated data, 
and the losses are printed for each epoch to monitor the training progress.
"""

# Training loop
for epoch in range(epochs):
    # Generate random noise
    noise = tf.random.normal(shape=(batch_size, latent_dim))

    # Generate fake samples and create a batch of real samples
    generated_data = generator(noise)
    real_data = x_train[np.random.choice(x_train.shape[0], batch_size, replace=False)]

    # Concatenate real and fake samples and create labels
    combined_data = tf.concat([real_data, generated_data], axis=0)
    labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

    # Train the discriminator
    discriminator_loss = discriminator.train_on_batch(combined_data, labels)

    # Train the generator (via GAN model)
    gan_loss = gan.train_on_batch(noise, tf.ones((batch_size, 1)))

    # Print the losses
    print(f"Epoch: {epoch+1}, Disc Loss: {discriminator_loss}, GAN Loss: {gan_loss}")



# Transformers and Autoregressive Models

"""
Autoregressive models, such as the GPT series, generate outputs sequentially, conditioning each step on previous outputs.
"""

""" It defines a Transformer model using the Keras Sequential API, which includes an embedding layer, a Transformer layer, 
and a dense layer with a softmax activation. 
This model is designed for tasks such as sequence-to-sequence language translation or natural language processing"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the Transformer model
transformer = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    layers.Transformer(num_layers, d_model, num_heads, dff,
        input_vocab_size=vocab_size, maximum_position_encoding=max_seq_length),
    layers.Dense(output_vocab_size, activation="softmax")
])
