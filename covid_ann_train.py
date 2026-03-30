#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow import keras
from tensorflow.keras import layers

# =========================================================
# 1) AYARLAR
# =========================================================
import kagglehub
DATASET_DIR = Path(kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")) / "COVID-19_Radiography_Dataset"

COVID_DIR = DATASET_DIR / "COVID" / "images"
NORMAL_DIR = DATASET_DIR / "Normal" / "images"

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# 2) YARDIMCI FONKSİYONLAR
# =========================================================
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def get_sorted_images(folder: Path):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda x: x.name)

def load_image(path: Path, img_size=(64, 64)):
    img = tf.keras.utils.load_img(path, color_mode="grayscale", target_size=img_size)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return arr.astype("float32")

def tr_label(idx: int) -> str:
    return "🔴 COVID" if int(idx) == 1 else "🟢 NORMAL"

def split_sequential(files, train_ratio=0.70):
    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    return train_files, test_files

def split_train_val_sequential(files, val_ratio=0.10):
    split_idx = int(len(files) * (1 - val_ratio))
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    return train_files, val_files

def build_xy(file_list, label, img_size):
    X = np.array([load_image(p, img_size) for p in file_list], dtype="float32")
    y = np.full(len(file_list), label, dtype="int32")
    return X, y

def print_last_file_examples(files, title, count=3):
    print(title)
    for p in files[-count:]:
        print("  ", p.name)
    print()

def print_sample_predictions(model, X, y, file_paths, n=6, threshold=0.5):
    if len(X) == 0:
        print("Örnek tahmin gösterilemedi. Test verisi boş.")
        return

    idxs = random.sample(range(len(X)), k=min(n, len(X)))

    print("=" * 70)
    print("🔎 ÖRNEK TAHMİNLER")
    print("=" * 70)
    print()

    for i, idx in enumerate(idxs, start=1):
        prob = float(model.predict(X[idx:idx+1], verbose=0)[0][0])
        pred = 1 if prob >= threshold else 0
        true = int(y[idx])
        correct = pred == true

        print(f"Örnek {i}:")
        print(f"  Dosya: {file_paths[idx].name}")
        print(f"  Tahmin Olasılığı (COVID): {prob:.6f} ({prob * 100:.4f}%)")
        print(f"  Tahmin: {tr_label(pred)}")
        print(f"  Gerçek: {tr_label(true)}")
        print(f"  Sonuç: {'✅ DOĞRU' if correct else '❌ YANLIŞ'}")
        print()

def show_probability_summary(y_prob):
    print("=" * 70)
    print("OLASILIK ÖZETİ")
    print("=" * 70)
    print(f"Min prob   : {y_prob.min():.6f}")
    print(f"Maks prob  : {y_prob.max():.6f}")
    print(f"Ortalama   : {y_prob.mean():.6f}")
    print(f"Medyan     : {np.median(y_prob):.6f}")
    print()

# =========================================================
# 3) DOSYALARI OKU
# =========================================================
print("=" * 70)
print("DOSYALAR OKUNUYOR")
print("=" * 70)

covid_files = get_sorted_images(COVID_DIR)
normal_files = get_sorted_images(NORMAL_DIR)

print(f"COVID toplam görüntü : {len(covid_files)}")
print(f"NORMAL toplam görüntü: {len(normal_files)}")
print()

# =========================================================
# 4) SINIF BAZLI SIRALI %70 TRAIN / %30 TEST
# =========================================================
covid_train_all, covid_test = split_sequential(covid_files, train_ratio=0.70)
normal_train_all, normal_test = split_sequential(normal_files, train_ratio=0.70)

covid_train, covid_val = split_train_val_sequential(covid_train_all, val_ratio=0.10)
normal_train, normal_val = split_train_val_sequential(normal_train_all, val_ratio=0.10)

print("=" * 70)
print("SIRALI VERİ BÖLME")
print("=" * 70)
print(f"COVID train      : {len(covid_train)}")
print(f"COVID validation : {len(covid_val)}")
print(f"COVID test       : {len(covid_test)}")
print(f"NORMAL train     : {len(normal_train)}")
print(f"NORMAL validation: {len(normal_val)}")
print(f"NORMAL test      : {len(normal_test)}")
print()

print_last_file_examples(covid_test, "COVID test son dosyalar:")
print_last_file_examples(normal_test, "NORMAL test son dosyalar:")

# =========================================================
# 5) X/Y OLUŞTUR
# =========================================================
print("=" * 70)
print("GÖRSELLER YÜKLENİYOR")
print("=" * 70)

X_covid_train, y_covid_train = build_xy(covid_train, 1, IMG_SIZE)
X_normal_train, y_normal_train = build_xy(normal_train, 0, IMG_SIZE)

X_covid_val, y_covid_val = build_xy(covid_val, 1, IMG_SIZE)
X_normal_val, y_normal_val = build_xy(normal_val, 0, IMG_SIZE)

X_covid_test, y_covid_test = build_xy(covid_test, 1, IMG_SIZE)
X_normal_test, y_normal_test = build_xy(normal_test, 0, IMG_SIZE)

X_train = np.concatenate([X_covid_train, X_normal_train], axis=0)
y_train = np.concatenate([y_covid_train, y_normal_train], axis=0)

X_val = np.concatenate([X_covid_val, X_normal_val], axis=0)
y_val = np.concatenate([y_covid_val, y_normal_val], axis=0)

X_test = np.concatenate([X_covid_test, X_normal_test], axis=0)
y_test = np.concatenate([y_covid_test, y_normal_test], axis=0)

test_file_paths = covid_test + normal_test

train_perm = np.random.permutation(len(X_train))
val_perm = np.random.permutation(len(X_val))

X_train, y_train = X_train[train_perm], y_train[train_perm]
X_val, y_val = X_val[val_perm], y_val[val_perm]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape  :", X_val.shape)
print("y_val shape  :", y_val.shape)
print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)
print()

print("=" * 70)
print("SINIF DAĞILIMI")
print("=" * 70)
print(f"Train NORMAL: {(y_train == 0).sum()} | COVID: {(y_train == 1).sum()}")
print(f"Val   NORMAL: {(y_val == 0).sum()} | COVID: {(y_val == 1).sum()}")
print(f"Test  NORMAL: {(y_test == 0).sum()} | COVID: {(y_test == 1).sum()}")
print()

# =========================================================
# 6) DATASET
# =========================================================
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(
    buffer_size=len(X_train), seed=SEED
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(
    BATCH_SIZE
).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
    BATCH_SIZE
).prefetch(tf.data.AUTOTUNE)

# =========================================================
# 7) CLASS WEIGHT
# =========================================================
normal_count = int((y_train == 0).sum())
covid_count = int((y_train == 1).sum())
total_train = len(y_train)

class_weight = {
    0: total_train / (2.0 * normal_count),
    1: total_train / (2.0 * covid_count),
}

print("=" * 70)
print("CLASS WEIGHT")
print("=" * 70)
print(class_weight)
print()

# =========================================================
# 8) ANN MODELİ
# =========================================================
print("=" * 70)
print("MODEL")
print("=" * 70)

model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), name="Giris"),
    layers.Flatten(name="Flatten"),
    layers.Dense(256, activation="relu", name="Gizli_1"),
    layers.Dropout(0.30, name="Dropout_1"),
    layers.Dense(64, activation="relu", name="Gizli_2"),
    layers.Dropout(0.20, name="Dropout_2"),
    layers.Dense(1, activation="sigmoid", name="Cikis"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]
)

model.summary()
print()

# =========================================================
# 9) EĞİTİM
# =========================================================
print("=" * 70)
print("EĞİTİM BAŞLIYOR")
print("=" * 70)
EPOCHS = 50

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[reduce_lr],
    class_weight=class_weight,
    verbose=1
)

# =========================================================
# 10) MODELİ KAYDET
# =========================================================
MODEL_SAVE_PATH = "covid_ann_model.keras"

model.save(MODEL_SAVE_PATH)
print(f"\n✅ Model kaydedildi: {MODEL_SAVE_PATH}")
# =========================================================
# 11) GRAFİKLER
# =========================================================
plt.figure(figsize=(7, 4))
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.title("Accuracy Grafiği")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(STATIC_DIR / "accuracy.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Loss Grafiği")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(STATIC_DIR / "loss.png", dpi=150)
plt.close()

# =========================================================
# 12) TEST SONUÇLARI
# =========================================================
print()
print("=" * 70)
print("TEST SONUÇLARI")
print("=" * 70)

test_metrics = model.evaluate(test_ds, verbose=0)
metric_names = model.metrics_names
for name, value in zip(metric_names, test_metrics):
    print(f"{name:12s}: {value:.4f}")
print()

y_prob = model.predict(X_test, verbose=0).ravel()
show_probability_summary(y_prob)

y_pred = (y_prob >= 0.5).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["NORMAL", "COVID"], zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print()

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.xticks([0, 1], ["NORMAL", "COVID"])
plt.yticks([0, 1], ["NORMAL", "COVID"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(STATIC_DIR / "confusion_matrix.png", dpi=150)
plt.close()

auc_score = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)

print(f"ROC-AUC: {auc_score:.4f}")

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], "--")
plt.title("ROC Eğrisi")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(STATIC_DIR / "roc_curve.png", dpi=150)
plt.close()

# =========================================================
# 13) ÖRNEK TAHMİNLER
# =========================================================
print()
print_sample_predictions(model, X_test, y_test, test_file_paths, n=6, threshold=0.5)
