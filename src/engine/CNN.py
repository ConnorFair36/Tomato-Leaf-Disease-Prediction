import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

#paths to data
train_path = r"C:\Users\sageg\Desktop\VCU\421 Stat\statgroupproj\Tomato-Leaf-Disease-Prediction\data\train"
val_path = r'c:\Users\sageg\Desktop\VCU\421 Stat\statgroupproj\Tomato-Leaf-Disease-Prediction\data\val'      

#normalize pixels in [0,1] range  
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

#training data 
train_generator = datagen.flow_from_directory(
    train_path,
    target_size=(256, 256),  #size of images from article
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',   #convert to grayscale
    shuffle=True
)

#testing data
validation_generator = datagen.flow_from_directory(
    val_path,
    target_size=(256, 256),  #size stated in article 
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',   #images to grayscale
    shuffle=False
)

#class info
class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)
print(f"\nClasses found ({num_classes} total):")
for i, class_name in enumerate(class_names):
    print(f"  {i}: {class_name}")

#CNN model
model = tf.keras.Sequential([
    #input layer for grayscale images
    tf.keras.layers.Input(shape=(256, 256, 1)),
    #1st convolutional layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    #2nd convolutional layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
   #3rd convolutional layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    #4th convolutional layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    #5th convolutional layer
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    #6th convolutional layer
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    #7th convolutional layer
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    #8th convolutional layer
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    #flatten & dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),  #attempt reduce overfitting
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

#compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'] #for MCC would need to implement manually
)

#print model architecture
print("\nModel Architecture:")
model.summary()

#train module
history = model.fit(
    train_generator,
    epochs=15, #can reduce epochs for meeting if we wanna run and see outputs quicker!
    validation_data=validation_generator,
    verbose=1
)

#save trained model - can do tensorflow folder method if thats preferred?
model.save('tomato_leaf_disease_model.h5')
print("\nModel saved as 'tomato_leaf_disease_model.h5'")

#evaluate on validation set
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"\nValidation Accuracy: {val_accuracy:.2%}")
print(f"Validation Loss: {val_loss:.4f}")