# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# **Step 1: Load Dataset**
# Define the image folder path and load the CSV
image_folder = "/content/bone-marrow-cell-classification/data/bone_marrow_cell_dataset/images"
csv_path = "/content/bone-marrow-cell-classification/abbreviations.csv"

# Load the CSV and inspect it
# Change the delimiter to ',' 
df = pd.read_csv(csv_path, delimiter=',')  
print(df.head())

# Add the full image path to the DataFrame
df['image_path'] = df['Abbreviation'].apply(lambda x: os.path.join(image_folder, x + '.png'))

# Ensure that all files exist
df = df[df['image_path'].apply(lambda x: os.path.exists(x))]
print(f"Dataset size after filtering invalid paths: {len(df)}")

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Category'])  # Assuming 'Category' contains class labels

# **Step 2: Train-Test Split**
X = df['image_path'].values
y = df['label'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# **Step 3: Load and Preprocess Images**
def load_and_preprocess_images(image_paths, target_size=(150, 150)):
    images = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Error: Image file not found: {img_path}")
            continue  # Skip missing images
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        images.append(img_array)
    return np.array(images)

# Load datasets
X_train_images = load_and_preprocess_images(X_train)
X_val_images = load_and_preprocess_images(X_val)
X_test_images = load_and_preprocess_images(X_test)

# Convert labels to one-hot encoding
from tensorflow.keras.utils import to_categorical
num_classes = len(label_encoder.classes_)
y_train_onehot = to_categorical(y_train, num_classes=num_classes)
y_val_onehot = to_categorical(y_val, num_classes=num_classes)
y_test_onehot = to_categorical(y_test, num_classes=num_classes)

# **Step 4: Define Models**
def create_custom_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_resnet50(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    for layer in base_model.layers:
        layer.trainable = False  # Freeze base layers
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# **Step 5: Train and Evaluate Models**
# Custom CNN
custom_cnn = create_custom_cnn((150, 150, 3))
custom_cnn_history = custom_cnn.fit(
    X_train_images, y_train_onehot,
    epochs=10,
    validation_data=(X_val_images, y_val_onehot),
    batch_size=32
)

# ResNet50
resnet50 = create_resnet50((150, 150, 3))
resnet50_history = resnet50.fit(
    X_train_images, y_train_onehot,
    epochs=10,
    validation_data=(X_val_images, y_val_onehot),
    batch_size=32
)

# **Step 6: Model Evaluation**
# Custom CNN
y_pred_custom = custom_cnn.predict(X_test_images)
y_pred_custom_classes = np.argmax(y_pred_custom, axis=1)

# ResNet50
y_pred_resnet = resnet50.predict(X_test_images)
y_pred_resnet_classes = np.argmax(y_pred_resnet, axis=1)

# **Metrics**
print("Custom CNN Classification Report:")
print(classification_report(y_test, y_pred_custom_classes))

print("ResNet50 Classification Report:")
print(classification_report(y_test, y_pred_resnet_classes))

# Confusion Matrices
conf_matrix_custom = confusion_matrix(y_test, y_pred_custom_classes)
conf_matrix_resnet = confusion_matrix(y_test, y_pred_resnet_classes)

# Plot confusion matrix for Custom CNN
sns.heatmap(conf_matrix_custom, annot=True, fmt="d", cmap="Blues")
plt.title("Custom CNN Confusion Matrix")
plt.show()

# Plot confusion matrix for ResNet50
sns.heatmap(conf_matrix_resnet, annot=True, fmt="d", cmap="Blues")
plt.title("ResNet50 Confusion Matrix")
plt.show()

# AUC-ROC
auc_custom = roc_auc_score(y_test_onehot, y_pred_custom, multi_class='ovr')
auc_resnet = roc_auc_score(y_test_onehot, y_pred_resnet, multi_class='ovr')

print(f"Custom CNN AUC-ROC: {auc_custom}")
print(f"ResNet50 AUC-ROC: {auc_resnet}")

# **Comparison Table**
results = {
    'Model': ['Custom CNN', 'ResNet50'],
    'Accuracy': [custom_cnn_history.history['val_accuracy'][-1], resnet50_history.history['val_accuracy'][-1]],
    'AUC-ROC': [auc_custom, auc_resnet]
}

results_df = pd.DataFrame(results)
print(results_df)
