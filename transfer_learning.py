import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

# Função para baixar imagens e criar dataset
def baixar_imagens(urls, pasta_destino):
    os.makedirs(pasta_destino, exist_ok=True)
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                caminho_arquivo = os.path.join(pasta_destino, f"imagem_{i}.jpg")
                with open(caminho_arquivo, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Erro ao baixar {url}")
        except Exception as e:
            print(f"Erro ao processar {url}: {e}")

# URLs de exemplo para imagens
urls_classe1 = [
    "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
    "https://upload.wikimedia.org/wikipedia/commons/6/62/Nokota_Horses_cropped.jpg"
]
urls_classe2 = [
    "https://upload.wikimedia.org/wikipedia/commons/e/e4/Gymnogyps_californianus_-San_Diego_Zoo-8a-4c.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/d9/Collage_of_Six_Cats-02.jpg"
]

# Baixar imagens para treino e validação
baixar_imagens(urls_classe1, "dataset/train/classe1")
baixar_imagens(urls_classe2, "dataset/train/classe2")
baixar_imagens(urls_classe1, "dataset/validation/classe1")
baixar_imagens(urls_classe2, "dataset/validation/classe2")

# Diretórios
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# Pré-processamento
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Modelo base (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compilar modelo
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treinar modelo
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator)
)

# Fine-tuning (ajuste fino)
for layer in base_model.layers[-10:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine_tune = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator)
)

# Salvar modelo
model.save('modelo_transfer_learning.h5')

# Visualizar resultados
plt.plot(history.history['accuracy'], label='Acurácia Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.title('Acurácia por Época')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Perda Treino')
plt.plot(history.history['val_loss'], label='Perda Validação')
plt.title('Perda por Época')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Função para prever imagens
def prever_imagem(caminho_imagem, modelo, class_indices):
    img = image.load_img(caminho_imagem, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = modelo.predict(img_array)
    class_index = np.argmax(predictions)
    classes = {v: k for k, v in class_indices.items()}
    return classes[class_index]

# Exemplo de previsão
caminho_imagem = 'dataset/validation/classe1/imagem_0.jpg'
class_indices = train_generator.class_indices
classe_prevista = prever_imagem(caminho_imagem, model, class_indices)
print(f"Classe prevista: {classe_prevista}")
