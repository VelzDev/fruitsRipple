import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# Папки для данных
train_dir = 'dataset/apple/train'
validation_dir = 'dataset/apple/validation'

# Настроим генераторы данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],  # вот тебе игра с яркостью
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,  # ResNet любит свой препроцесс
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

val_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Загружаем ResNet50 без верхнего слоя
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Замораживаем слои ResNet
base_model.trainable = False

# Строим модель
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 класса
])

# Компиляция модели с уменьшенным learning rate
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучаем модель
model.fit(train_generator, epochs=30, validation_data=val_generator)

# Сохраняем модель
model.save('apple_classifier-demo.h5')
print("✅ Модель сохранена в apple_classifier-demo.h5")