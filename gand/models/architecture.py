from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization
from keras.layers import Input, Embedding, Reshape, LeakyReLU, Concatenate, Conv2DTranspose
from keras.optimizers import Adam

input_shape = (28, 28, 1)

# ----------------
# Deep Model -----
# ----------------
def deep_model(img_shape=(28, 28, 1),
               classes=10):
    model = Sequential([
        Conv2D(filters=32, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding='same',
               input_shape=img_shape),        
        Conv2D(filters=32, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Dropout(0.2),

        Conv2D(filters=64, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding='same'),
        Conv2D(filters=64, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Dropout(0.2),

        Conv2D(filters=128, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding='same'),
        Conv2D(filters=128, kernel_size=3, activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Dropout(0.2),

        Flatten(),
        Dense(128, activation='relu', kernel_initializer='he_uniform'),
        Dropout(0.5),

        Dense(classes, activation='softmax')
    ], name='deep_model_stable')

    return model

# deep_model_cifar10().summary()28

# ----------------
# Baseline 1: VGG
# ----------------
def baseline_1(name: str = 'baseline_1') -> type(Sequential):
    model = Sequential([
        Conv2D(filters=32, kernel_size=3,
               activation="relu",
               input_shape=input_shape, kernel_initializer='he_uniform'),
        Conv2D(32, 3, activation="relu", kernel_initializer='he_uniform'),
        MaxPooling2D(2),

        Flatten(),
        Dense(128, kernel_initializer='he_uniform'),
        Dense(10, activation="softmax"),
    ], name=name)

    return model


# ----------------
# Baseline 2: VGG
# ----------------
def baseline_2(name: str = 'baseline_2') -> type(Sequential):
    model = Sequential([
        Conv2D(filters=32, kernel_size=3,
               activation="relu",
               input_shape=input_shape, kernel_initializer='he_uniform'),
        Conv2D(32, 3, activation="relu", kernel_initializer='he_uniform'),
        MaxPooling2D(2),

        Conv2D(64, 3, activation="relu", kernel_initializer='he_uniform'),
        Conv2D(64, 3, activation="relu", kernel_initializer='he_uniform'),
        MaxPooling2D(2),

        Flatten(),
        Dense(128, kernel_initializer='he_uniform'),
        Dense(10, activation="softmax"),
    ], name=name)

    return model


# ----------------
# Baseline 3: VGG
# ----------------
def baseline_3(name: str = 'baseline_3') -> type(Sequential):
    model = Sequential([
        Conv2D(filters=32, kernel_size=3,
               activation="relu",
               input_shape=input_shape, padding='same', kernel_initializer='he_uniform'),
        Conv2D(32, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        MaxPooling2D(2),

        Conv2D(64, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        Conv2D(64, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        MaxPooling2D(2),

        Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        MaxPooling2D(2),

        Flatten(),
        Dense(128, kernel_initializer='he_uniform'),
        Dense(10, activation="softmax"),
    ], name=name)

    return model


# -----------------------
# Dropout Regularisation
# -----------------------
def dropout_regularisation(
                           name: str = 'dropout_regularisation') -> type(Sequential):
    model = Sequential([
        Conv2D(filters=32, kernel_size=3,
               activation="relu",
               input_shape=input_shape, padding='same', kernel_initializer='he_uniform'),
        Conv2D(32, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        MaxPooling2D(2),
        Dropout(0.2),

        Conv2D(64, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        Conv2D(64, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        MaxPooling2D(2),
        Dropout(0.2),

        Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        MaxPooling2D(2),
        Dropout(0.2),

        Flatten(),
        Dense(128, kernel_initializer='he_uniform'),
        Dropout(0.2),
        Dense(10, activation="softmax"),
    ], name=name)

    return model


# ------------------------------------	
# Variational Dropout Regularisation
# ------------------------------------
def variational_dropout_regularisation(name: str = 'variational_dropout_regularisation') -> type(Sequential):
    model = Sequential([
        Conv2D(filters=32, kernel_size=3,
               activation="relu", padding='same',
               input_shape=input_shape, kernel_initializer='he_uniform'),
        Conv2D(32, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        MaxPooling2D(2),
        Dropout(0.2),

        Conv2D(64, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        Conv2D(64, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        MaxPooling2D(2),
        Dropout(0.3),

        Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        Conv2D(128, 3, activation="relu", padding='same', kernel_initializer='he_uniform'),
        MaxPooling2D(2),
        Dropout(0.4),

        Flatten(),
        Dense(128, kernel_initializer='he_uniform'),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ], name=name)

    return model


# -----------
# Batch Norm
# -----------
def batch_norm(name: str = 'batch_norm') -> type(Sequential):
    model = Sequential([
        Conv2D(filters=32, kernel_size=3,
               activation="relu", padding='same',
               input_shape=input_shape),
        Conv2D(32, 3, activation="relu", padding='same'),
        BatchNormalization(axis=-1),
        MaxPooling2D(2),

        Conv2D(64, 3, activation="relu", padding='same'),
        Conv2D(64, 3, activation="relu", padding='same'),
        BatchNormalization(axis=-1),
        MaxPooling2D(2),

        Conv2D(128, 3, activation="relu", padding='same'),
        Conv2D(128, 3, activation="relu", padding='same'),
        BatchNormalization(axis=-1),
        MaxPooling2D(2),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ], name=name)

    return model


# ------------------------------------------------------------
# Variational Dropout Regularisation With Batch Normalisation
# ------------------------------------------------------------
def variational_batch_norm(name: str = 'variational_batch_norm') -> type(Sequential):
    model = Sequential([
        Conv2D(filters=32, kernel_size=3,
               activation="relu", padding='same',
               input_shape=input_shape),
        Conv2D(32, 3, activation="relu", padding='same'),
        BatchNormalization(axis=-1),
        MaxPooling2D(2),
        Dropout(0.2),

        Conv2D(64, 3, activation="relu", padding='same'),
        Conv2D(64, 3, activation="relu", padding='same'),
        BatchNormalization(axis=-1),
        MaxPooling2D(2),
        Dropout(0.3),

        Conv2D(128, 3, activation="relu", padding='same'),
        Conv2D(128, 3, activation="relu", padding='same'),
        BatchNormalization(axis=-1),
        MaxPooling2D(2),
        Dropout(0.4),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ], name=name)

    return model


# --------------
# DISCRIMINATOR
# --------------
def discriminator(in_shape=(28, 28, 1), n_classes=10):
    # LABEL INPUT
    in_label = Input(shape=(1,))

    # EMBEDDING FOR CATEGORICAL INPUT / EACH LABEL REPRESENTED BY A VECTOR OF SIZE 50
    li = Embedding(n_classes, 50)(in_label)

    n_nodes = in_shape[0] * in_shape[1] * in_shape[2]
    li = Dense(n_nodes)(li)

    li = Reshape((in_shape[0], in_shape[1], in_shape[2]))(li)

    in_image = Input(shape=in_shape)  # IMAGE INPUT : 32x32x3
    merge = Concatenate()([in_image, li])  # MERGED INPUT: 32x32x4

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)

    fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)

    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)

    out_layer = Dense(1, activation='sigmoid')(fe)

    # COMBINE INPUT IMAGE AND LABEL
    model = Model([in_image, in_label], out_layer, name='discriminator')

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

    return model


# ----------
# GENERATOR
# ----------
def define_generator(latent_dim, n_classes=10):
    # LABEL INPUT
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)

    n_nodes = 7 * 7
    li = Dense(n_nodes)(li)
    li = Reshape((7, 7, 1))(li)

    in_lat = Input(shape=(latent_dim,))

    n_nodes = 128 * 7 * 7
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7, 7, 128))(gen)

    merge = Concatenate()([gen, li])

    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    out_layer = Conv2D(1, (7, 7), activation='tanh', padding='same')(gen)

    model = Model([in_lat, in_label], out_layer, name='generator')

    return model


# ----------
# GAN MODEL
# ----------
def define_gan(g_model, d_model):
    d_model.trainable = False

    # NOISE AND LABEL INPUTS FROM THE MODEL
    gen_noise, gen_label = g_model.input

    # GENERATOR OUTPUT
    gen_output = g_model.output

    gan_output = d_model([gen_output, gen_label])

    model = Model([gen_noise, gen_label], gan_output, name='gan')

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
