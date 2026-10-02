"""
Deep Learning Assignment 2: MNIST Classification & Regularization Experiments
Author: Yusuf Imantaka Bastari (24/532731/PA/22530)
Department of Computer Science and Electronics, FMIPA UGM

Experiments:
- Baseline: 1 hidden layer Dense(64, relu), Softmax(10), Adam, Categorical Crossentropy
- Scenario 2(a): L1 Regularization
- Scenario 2(b): L2 Regularization
- Scenario 2(c): Dropout
- Scenario 2(d): Early Stopping
- Scenario 2(e): L1 Regularization + Dropout
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, regularizers, callbacks
from keras.datasets import mnist

# Ensure reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Set up output directories
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. Load and preprocess MNIST dataset (matching slides 14-23)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

print(f"Dataset loaded: X_train: {X_train.shape}, X_test: {X_test.shape}")

# Hyperparameter definitions with data-driven justifications:
# L1 rate: 1e-4 provides sparse weight selection without underfitting
# L2 rate: 1e-4 standard weight decay stabilizing L2 norm
# Dropout rate: 0.2 retains sufficient capacity for 64 hidden units
# Early stopping: patience=3, monitor='val_loss', restore_best_weights=True
L1_RATE = 1e-4
L2_RATE = 1e-4
DROPOUT_RATE = 0.2
ES_PATIENCE = 3

results = {}

def plot_loss(history, title, filename, stopped_epoch=None):
    """Generate high quality loss curve matching slide placement."""
    plt.figure(figsize=(5.5, 3.5), dpi=300)
    epochs_range = range(1, len(history.history["loss"]) + 1)
    
    plt.plot(epochs_range, history.history["loss"], "r-o", markersize=4, label="Training Loss", linewidth=1.5)
    plt.plot(epochs_range, history.history["val_loss"], "b-s", markersize=4, label="Validation Loss", linewidth=1.5)
    
    if stopped_epoch is not None:
        plt.axvline(x=stopped_epoch, color="green", linestyle="--", label=f"Early Stop (Epoch {stopped_epoch})")
        
    plt.title(title, fontsize=11, fontweight="bold", pad=8)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss (Categorical Crossentropy)", fontsize=10)
    plt.legend(frameon=True, fontsize=9, loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}", dpi=300)
    plt.close()
    print(f"Saved figure: figures/{filename}")

# --- Baseline ---
print("\n=== Running Baseline ===")
model_base = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model_base.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
hist_base = model_base.fit(X_train, y_train_cat, epochs=10, batch_size=100, validation_data=(X_test, y_test_cat), verbose=1)
eval_base = model_base.evaluate(X_test, y_test_cat, verbose=0)
plot_loss(hist_base, "Baseline MLP: Training vs Validation Loss", "loss_baseline.png")
results["baseline"] = {
    "name": "Baseline MLP",
    "params": model_base.count_params(),
    "train_loss": hist_base.history["loss"][-1],
    "train_acc": hist_base.history["acc"][-1],
    "val_loss": hist_base.history["val_loss"][-1],
    "val_acc": hist_base.history["val_acc"][-1],
    "test_loss": eval_base[0],
    "test_acc": eval_base[1]
}

# --- Scenario 2(a): L1 Regularization ---
print("\n=== Scenario 2(a): L1 Regularization ===")
model_l1 = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l1(L1_RATE)),
    layers.Dense(10, activation="softmax")
])
model_l1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
hist_l1 = model_l1.fit(X_train, y_train_cat, epochs=10, batch_size=100, validation_data=(X_test, y_test_cat), verbose=1)
eval_l1 = model_l1.evaluate(X_test, y_test_cat, verbose=0)
plot_loss(hist_l1, f"Scenario 2(a): L1 Regularization (rate = {L1_RATE})", "loss_l1.png")
results["scenario_2a"] = {
    "name": f"L1 Regularization (lambda={L1_RATE})",
    "rate": L1_RATE,
    "params": model_l1.count_params(),
    "train_loss": hist_l1.history["loss"][-1],
    "train_acc": hist_l1.history["acc"][-1],
    "val_loss": hist_l1.history["val_loss"][-1],
    "val_acc": hist_l1.history["val_acc"][-1],
    "test_loss": eval_l1[0],
    "test_acc": eval_l1[1]
}

# --- Scenario 2(b): L2 Regularization ---
print("\n=== Scenario 2(b): L2 Regularization ===")
model_l2 = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(L2_RATE)),
    layers.Dense(10, activation="softmax")
])
model_l2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
hist_l2 = model_l2.fit(X_train, y_train_cat, epochs=10, batch_size=100, validation_data=(X_test, y_test_cat), verbose=1)
eval_l2 = model_l2.evaluate(X_test, y_test_cat, verbose=0)
plot_loss(hist_l2, f"Scenario 2(b): L2 Regularization (rate = {L2_RATE})", "loss_l2.png")
results["scenario_2b"] = {
    "name": f"L2 Regularization (lambda={L2_RATE})",
    "rate": L2_RATE,
    "params": model_l2.count_params(),
    "train_loss": hist_l2.history["loss"][-1],
    "train_acc": hist_l2.history["acc"][-1],
    "val_loss": hist_l2.history["val_loss"][-1],
    "val_acc": hist_l2.history["val_acc"][-1],
    "test_loss": eval_l2[0],
    "test_acc": eval_l2[1]
}

# --- Scenario 2(c): Dropout ---
print("\n=== Scenario 2(c): Dropout ===")
model_do = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(DROPOUT_RATE),
    layers.Dense(10, activation="softmax")
])
model_do.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
hist_do = model_do.fit(X_train, y_train_cat, epochs=10, batch_size=100, validation_data=(X_test, y_test_cat), verbose=1)
eval_do = model_do.evaluate(X_test, y_test_cat, verbose=0)
plot_loss(hist_do, f"Scenario 2(c): Dropout (rate = {DROPOUT_RATE})", "loss_dropout.png")
results["scenario_2c"] = {
    "name": f"Dropout (p={DROPOUT_RATE})",
    "rate": DROPOUT_RATE,
    "params": model_do.count_params(),
    "train_loss": hist_do.history["loss"][-1],
    "train_acc": hist_do.history["acc"][-1],
    "val_loss": hist_do.history["val_loss"][-1],
    "val_acc": hist_do.history["val_acc"][-1],
    "test_loss": eval_do[0],
    "test_acc": eval_do[1]
}

# --- Scenario 2(d): Early Stopping ---
print("\n=== Scenario 2(d): Early Stopping ===")
model_es = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model_es.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
es_callback = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=ES_PATIENCE,
    restore_best_weights=True,
    verbose=1
)
hist_es = model_es.fit(
    X_train, y_train_cat,
    epochs=25,
    batch_size=100,
    validation_data=(X_test, y_test_cat),
    callbacks=[es_callback],
    verbose=1
)
stopped_epoch = es_callback.stopped_epoch + 1 if es_callback.stopped_epoch > 0 else len(hist_es.history["loss"])
best_epoch = np.argmin(hist_es.history["val_loss"]) + 1
eval_es = model_es.evaluate(X_test, y_test_cat, verbose=0)
plot_loss(hist_es, f"Scenario 2(d): Early Stopping (Patience = {ES_PATIENCE})", "loss_early_stopping.png", stopped_epoch=stopped_epoch)
results["scenario_2d"] = {
    "name": f"Early Stopping (patience={ES_PATIENCE})",
    "patience": ES_PATIENCE,
    "stopped_epoch": int(stopped_epoch),
    "best_epoch": int(best_epoch),
    "params": model_es.count_params(),
    "train_loss": hist_es.history["loss"][best_epoch - 1],
    "train_acc": hist_es.history["acc"][best_epoch - 1],
    "val_loss": hist_es.history["val_loss"][best_epoch - 1],
    "val_acc": hist_es.history["val_acc"][best_epoch - 1],
    "test_loss": eval_es[0],
    "test_acc": eval_es[1]
}

# --- Scenario 2(e): L1 Regularization + Dropout ---
print("\n=== Scenario 2(e): L1 Regularization + Dropout ===")
model_l1_do = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l1(L1_RATE)),
    layers.Dropout(DROPOUT_RATE),
    layers.Dense(10, activation="softmax")
])
model_l1_do.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
hist_l1_do = model_l1_do.fit(X_train, y_train_cat, epochs=10, batch_size=100, validation_data=(X_test, y_test_cat), verbose=1)
eval_l1_do = model_l1_do.evaluate(X_test, y_test_cat, verbose=0)
plot_loss(hist_l1_do, f"Scenario 2(e): L1 ({L1_RATE}) + Dropout ({DROPOUT_RATE})", "loss_l1_dropout.png")
results["scenario_2e"] = {
    "name": f"L1 ({L1_RATE}) + Dropout ({DROPOUT_RATE})",
    "l1_rate": L1_RATE,
    "dropout_rate": DROPOUT_RATE,
    "params": model_l1_do.count_params(),
    "train_loss": hist_l1_do.history["loss"][-1],
    "train_acc": hist_l1_do.history["acc"][-1],
    "val_loss": hist_l1_do.history["val_loss"][-1],
    "val_acc": hist_l1_do.history["val_acc"][-1],
    "test_loss": eval_l1_do[0],
    "test_acc": eval_l1_do[1]
}

# Save experiment summary
with open("experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Summary of All Experiments ===")
print(f"{'Scenario':<25} | {'Params':<8} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10}")
print("-" * 85)
for k, v in results.items():
    print(f"{v['name']:<25} | {v['params']:<8} | {v['train_loss']:<10.4f} | {v['train_acc']:<10.4f} | {v['val_loss']:<10.4f} | {v['val_acc']:<10.4f}")
