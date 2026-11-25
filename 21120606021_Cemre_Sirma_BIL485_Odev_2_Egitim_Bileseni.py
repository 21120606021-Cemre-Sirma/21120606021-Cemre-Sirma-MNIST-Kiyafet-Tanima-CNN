import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
CLASS_NAMES = ["T-shirt", "Pantolon", "Kazak", "Elbise", "Ceket","Sandalet", "Gömlek", "Spor Ayakkabı", "Çanta", "Yarım Bot"]
(XTF, YTF), (XT, YT) = tf.keras.datasets.fashion_mnist.load_data()
XTF = XTF.astype("float32") / 255.0
XT = XT.astype("float32") / 255.0
XTF = np.expand_dims(XTF, axis=-1)
XT = np.expand_dims(XT, axis=-1)
XT, XV, YT, YV = train_test_split(XTF, YTF, test_size=0.2, random_state=42, stratify=YTF)
print(f"Eğitim Seti: {XT.shape}, Doğrulama Seti: {XV.shape}, Test Seti: {XT.shape}")
model = models.Sequential([layers.Input(shape=(28, 28, 1)),layers.Conv2D(32, (3, 3), activation="relu", padding="same"),layers.MaxPooling2D((2, 2)),layers.Conv2D(64, (3, 3), activation="relu", padding="same"),layers.MaxPooling2D((2, 2)),layers.Flatten(),layers.Dropout(0.3),layers.Dense(128, activation="relu"),layers.Dense(10, activation="softmax")])
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.summary()
print("\nEğitim başlıyor...")
TS = time.time()
history = model.fit(XT, YT,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_data=(XV, YV))
TT = time.time() - TS
print(f"\nToplam Eğitim Süresi: {TT:.2f} saniye")
IS = time.time()
YPP = model.predict(XT, verbose=0)
ITT = time.time() - IS
YP = np.argmax(YPP, axis=1)
AITM = (ITT / len(XT)) * 1000
ACC = accuracy_score(YT, YP)
PREC = precision_score(YT, YP, average="macro")
REC = recall_score(YT, YP, average="macro")
F1 = f1_score(YT, YP, average="macro")
print("\n--- TEST PERFORMANS METRİKLERİ ---")
print(f"Accuracy  : {ACC:.4f}")
print(f"Precision : {PREC:.4f}")
print(f"Recall    : {REC:.4f}")
print(f"F1-Score  : {F1:.4f}")
print(f"Ortalama Tahmin Süresi (Inference Time): {AITM:.4f} ms / görüntü")
print("\nDetaylı Sınıflandırma Raporu:")
print(classification_report(YT, YP, target_names=CLASS_NAMES))
model.save("egitilmis_mlp_modeli.keras")
print("Model 'egitilmis_mlp_modeli.keras' olarak kaydedildi.")