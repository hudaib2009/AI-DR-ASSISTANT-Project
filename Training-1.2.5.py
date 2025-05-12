import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Add, Multiply, GlobalAveragePooling2D, Reshape, Attention
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                           precision_recall_curve, roc_curve, roc_auc_score,
                           average_precision_score, cohen_kappa_score,
                           matthews_corrcoef, balanced_accuracy_score,
                           jaccard_score, hamming_loss, recall_score)
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

def load_data(csv_path, img_size=(128, 128)):
    """
    Load data from CSV file for multi-class classification.
    
    Args:
        csv_path (str): Path to the CSV file containing image data
        img_size (tuple): Target size for the images (width, height)
    
    Returns:
        tuple: (images, labels, class_names) as numpy arrays and list
    """
    print("Loading data from CSV...")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get the label column (assuming it's the last column)
    labels = df.iloc[:, -1].values
    
    # Get unique class names and create mapping
    unique_classes = np.unique(labels)
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Convert string labels to indices
    numeric_labels = np.array([class_to_idx[label] for label in labels])
    
    # Get all columns except the last one (which is the label)
    image_columns = df.columns[:-1]
    
    # Convert image data to numpy array
    images = df[image_columns].values
    
    # Reshape the flattened images back to 3D (height, width, channels)
    num_images = len(images)
    images = images.reshape(num_images, img_size[0], img_size[1], 3)
    
    # Save class mapping to file
    with open('class_mapping.json', 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class
        }, f, indent=4)
    
    print(f"Loaded {num_images} images from CSV")
    print(f"Number of classes: {len(unique_classes)}")
    print("Classes:", unique_classes)
    return images, numeric_labels, unique_classes

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate the model and print detailed metrics.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: Test labels
        class_names: List of class names
    """
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("\n=== Detailed classification report ===")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate metrics for each class
    metrics = {}
    for i, class_name in enumerate(class_names):
        class_precision = precision_score(y_true_classes == i, y_pred_classes == i)
        class_recall = recall_score(y_true_classes == i, y_pred_classes == i)
        class_f1 = f1_score(y_true_classes == i, y_pred_classes == i)
        
        metrics[class_name] = {
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1
        }
    
    print("\n=== All metrics ===")
    print(f"Balanced Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"Cohen's Kappa: {metrics['kappa']:.4f}")
    print(f"Matthews Correlation Coefficient: {metrics['mcc']:.4f}")
    print(f"Jaccard Score: {metrics['jaccard']:.4f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    
    return metrics

def attention_block(x, filters):
    # Self-attention mechanism
    attention = Attention()([x, x])
    x = Add()([x, attention])
    return x

def create_model(input_shape=(128, 128, 1), num_classes=12):
    """
    Create a model for multi-class classification.
    
    Args:
        input_shape (tuple): Shape of input images
        num_classes (int): Number of output classes
    """
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
  
    # Block 2
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
  
    # Block 3
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
  
    # Optional: Attention here
    x = attention_block(x, 128)
  
    # Block 4
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
  
  # Head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, output)

def train_model(csv_path, total_subset=1, train_size=0.8, test_size=0.2):
    """
    Train the model using data from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing image data
        total_subset (float): Proportion of data to use (e.g., 0.1 for 10%)
        train_size (float): Proportion of data to use for training
        test_size (float): Proportion of data to use for testing
    """
    if train_size + test_size != 1.0:
        raise ValueError("train_size and test_size must sum to 1 when not using validation.")

    # Load data from CSV
    images, labels, class_names = load_data(csv_path)
    images, labels = shuffle(images, labels, random_state=42)

    # Convert labels to one-hot encoding
    num_classes = len(class_names)
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    # Take subset of the data
    subset_size = int(len(images) * total_subset)
    images, labels = images[:subset_size], labels[:subset_size]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=42
    )

    # Normalize pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Create the model
    model = create_model(num_classes=num_classes)
    
    model.compile(
        optimizer=Adam(learning_rate=0.00005, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall(), 
                tf.keras.metrics.F1Score()]
    )
    
    def lr_schedule(epoch):
        if epoch < 50:
            return 0.00005
        elif epoch < 100:
            return 0.00001
        else:
            return 0.000005
    
    # setup Callbacks
    callbacks = [
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        LearningRateScheduler(lr_schedule)
    ]
    
    # create the data generator 
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size = BATCHSIZE),
        # steps_per_epoch = len(X_train) // 256,
        #   It is not entirely accurate. We calculate the number of steps per epochs (steps_p-
        #   er_epochs) by taking the ceiling of the number of training samples (X_train) divi- 
        #   ded by Batch Size (batch_size) - not using floor division ( // ). So, if we want 
        #   to specify the number of steps, we can run {math.ceil(112000 / 256)}.
        #   NOTE: Using {len(X_train) // 256} is ok if we are ok with dropping some samples.
        #   That said, you can simply remove it and let the program automatically calculate it
        #   automatically.
        #   -Written by Ahmad Shatnawi
        epochs = EPOCHS,
        validation_data = (X_test, y_test),
        callbacks = callbacks
    )
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test, class_names)
    
    return model, history, metrics

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    csv_path = "images_Batched.csv"  # Path to your CSV file
    BATCHSIZE = 256
    EPOCHS = 200
    
    total_subset = input("Enter the total subset proportion (e.g., 0.05 for 5% of the data) [default 1]: ")
    total_subset = float(total_subset) if total_subset else 1

    train_size = input("Enter the train size proportion (e.g., 0.8 for 80% of subset) [default 0.8]: ")
    train_size = float(train_size) if train_size else 0.8

    test_size = input("Enter the test size proportion (e.g., 0.2 for 20% of subset) [default 0.2]: ")
    test_size = float(test_size) if test_size else 0.2
    
    model, history, metrics = train_model(csv_path, total_subset, train_size, test_size)
