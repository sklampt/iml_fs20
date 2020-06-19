import numpy as np
import pandas as pd
import tensorflow as tf
import random

data = pd.read_csv(
    'data/train_triplets.txt',
    dtype=str,
    sep=" ",
    header=None,
    names=['A','B','C'],
    # nrows=100,
)
data = data.apply(lambda x: x + ".jpg")
data['class'] = np.ones(len(data.index)).astype(int).astype(str)

# Randomly switch B and C -> class = 0
for index, row in data.iterrows():
    if random.choice([True, False]):
        data['B'][index], data['C'][index] = data['C'][index], data['B'][index]
        data['class'][index] = "0"

A_data = data[['A', 'class']].copy()
B_data = data[['B', 'class']].copy()
C_data = data[['C', 'class']].copy()
A_train = A_data.iloc[:50000,:]
B_train = B_data.iloc[:50000,:]
C_train = C_data.iloc[:50000,:]
A_test = A_data.iloc[50000:,:]
B_test = B_data.iloc[50000:,:]
C_test = C_data.iloc[50000:,:]
y = data['class']

def make_model(input_shape):
    x = tf.keras.layers.Conv2D(16, 3, activation='relu')(input_shape)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    model = tf.keras.Model(input_shape, x)
    return model

inputa = tf.keras.Input(shape=(450,300,3))
inputb = tf.keras.Input(shape=(450,300,3))
inputc = tf.keras.Input(shape=(450,300,3))
modela = make_model(inputa)
modelb = make_model(inputb)
modelc = make_model(inputc)
combined = tf.keras.layers.concatenate([modela.output, modelb.output,modelc.output])
end = tf.keras.layers.Dense(512, activation='relu')(combined)
output = tf.keras.layers.Dense(1, activation='sigmoid')(end)
model = tf.keras.Model(inputs=[modela.input, modelb.input,modelc.input], outputs=output)
model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    metrics=['acc'],
)

input_imgen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
test_imgen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

def generate_generator_multiple(generator, dir1, dir2, dir3, batch_size):
    genX1 = generator.flow_from_dataframe(
        dir1,
        directory=r'data/food',
        x_col="A",
        y_col="class",
        target_size=(450,300),
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False, 
        seed=7,
        validate_filenames=False,
        drop_duplicates=False,
    )
    
    genX2 = generator.flow_from_dataframe(
        dir2,
        directory=r'data/food',
        x_col="B",
        y_col="class",
        target_size=(450,300),
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False, 
        seed=7,
        validate_filenames=False,
        drop_duplicates=False,
    )

    genX3 = generator.flow_from_dataframe(
        dir3,
        directory=r'data/food',
        x_col="C",
        y_col="class",
        target_size=(450,300),
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False, 
        seed=7,
        validate_filenames=False,
        drop_duplicates=False,
    )

    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        yield [X1i[0], X2i[0],X3i[0]], X2i[1]  #Yield both images and their mutual label

inputgenerator = generate_generator_multiple(
    generator=input_imgen,
    dir1=A_train,
    dir2=B_train,
    dir3=C_train,
    batch_size=32,
)

testgenerator = generate_generator_multiple(
    test_imgen,
    dir1=A_test,
    dir2=B_test,
    dir3=C_test,
    batch_size=32,
)

history = model.fit_generator(
    inputgenerator,
    epochs=10,
    steps_per_epoch=100,
    validation_data=testgenerator,
    validation_steps=10,
    shuffle=False,
    verbose=1,
)