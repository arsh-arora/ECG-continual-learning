#!/usr/bin/env python
# coding: utf-8

# # Importing requirements

# In[1]:


import os
import numpy as np
import pandas as pd
import wfdb
import ast
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from collections import Counter
import time
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

DATA_PATH = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'

ptbxl_df = pd.read_csv(os.path.join(DATA_PATH, 'ptbxl_database.csv'))
scp_statements = pd.read_csv(os.path.join(DATA_PATH, 'scp_statements.csv'), index_col=0)

diagnostic_scps = scp_statements[scp_statements['diagnostic'] == 1].index.values

scp_to_superclass = scp_statements['diagnostic_class'].to_dict()
scp_to_subclass = scp_statements['diagnostic_subclass'].to_dict()


# In[4]:


ptbxl_df['scp_codes'] = ptbxl_df['scp_codes'].apply(lambda x: ast.literal_eval(x))


# In[5]:


def aggregate_diagnostic_labels(df, scp_codes, scp_to_agg):
    df = df.copy()
    def aggregate_labels(scp_codes_dict):
        labels = set()
        for code in scp_codes_dict.keys():
            if code in scp_codes:
                label = scp_to_agg.get(code)
                if label:
                    labels.add(label)
        return list(labels)
    df['diagnostic_labels'] = df['scp_codes'].apply(aggregate_labels)
    return df

ptbxl_df = aggregate_diagnostic_labels(ptbxl_df, diagnostic_scps, scp_to_superclass)
ptbxl_df = ptbxl_df.rename(columns={'diagnostic_labels': 'superclass_labels'})

ptbxl_df = aggregate_diagnostic_labels(ptbxl_df, diagnostic_scps, scp_to_subclass)
ptbxl_df = ptbxl_df.rename(columns={'diagnostic_labels': 'subclass_labels'})


# In[6]:


ptbxl_df = ptbxl_df[ptbxl_df['superclass_labels'].map(len) > 0]


# In[7]:


train_df = ptbxl_df[ptbxl_df.strat_fold <= 8]
val_df = ptbxl_df[ptbxl_df.strat_fold == 9]
test_df = ptbxl_df[ptbxl_df.strat_fold == 10]


# In[8]:


def load_data(df, sampling_rate, data_path):
    data = []
    i = 0
    if sampling_rate == 100:
        filenames = df['filename_lr'].values
    else:
        filenames = df['filename_hr'].values
    for filename in filenames:
        file_path = os.path.join(data_path, filename)
        signals, _ = wfdb.rdsamp(file_path)
        data.append(signals)
    return np.array(data)

X_train = load_data(train_df, sampling_rate=100, data_path=DATA_PATH)
X_val = load_data(val_df, sampling_rate=100, data_path=DATA_PATH)
X_test = load_data(test_df, sampling_rate=100, data_path=DATA_PATH)


# In[9]:


train_labels_super = train_df['superclass_labels'].values
val_labels_super = val_df['superclass_labels'].values
test_labels_super = test_df['superclass_labels'].values

mlb_super = MultiLabelBinarizer()
y_train_super = mlb_super.fit_transform(train_labels_super)
y_val_super = mlb_super.transform(val_labels_super)
y_test_super = mlb_super.transform(test_labels_super)
classes_super = mlb_super.classes_


# In[10]:


train_labels_sub = train_df['subclass_labels'].values
val_labels_sub = val_df['subclass_labels'].values
test_labels_sub = test_df['subclass_labels'].values

mlb_sub = MultiLabelBinarizer()
y_train_sub = mlb_sub.fit_transform(train_labels_sub)
y_val_sub = mlb_sub.transform(val_labels_sub)
y_test_sub = mlb_sub.transform(test_labels_sub)
classes_sub = mlb_sub.classes_


# In[11]:


def normalize_data_per_channel(X):
    X = np.transpose(X, (0, 2, 1))
    mean = np.mean(X, axis=(0, 2), keepdims=True)
    std = np.std(X, axis=(0, 2), keepdims=True)
    X = (X - mean) / std
    X = np.transpose(X, (0, 2, 1))
    return X

X_train = normalize_data_per_channel(X_train)
X_val = normalize_data_per_channel(X_val)
X_test = normalize_data_per_channel(X_test)


# In[12]:


class_counts_super = np.sum(y_train_super, axis=0)
total_samples_super = y_train_super.shape[0]

class_weight_super = {}
for i, count in enumerate(class_counts_super):
    class_weight_super[i] = total_samples_super / (len(class_counts_super) * count)

class_counts_sub = np.sum(y_train_sub, axis=0)
total_samples_sub = y_train_sub.shape[0]

class_weight_sub = {}
for i, count in enumerate(class_counts_sub):
    class_weight_sub[i] = total_samples_sub / (len(class_counts_sub) * count)


# In[13]:


num_classes_super = y_train_super.shape[1]
class_totals = np.sum(y_train_super, axis=0)
class_weights = class_totals.max() / class_totals
weights_array = np.array(class_weights, dtype=np.float32)


# In[14]:


num_classes_sub = y_train_sub.shape[1]
class_totals_sub = np.sum(y_train_sub, axis=0)
class_weights_sub = class_totals_sub.max() / class_totals_sub
weights_array_sub = np.array(class_weights_sub, dtype=np.float32)


# In[15]:


y_train_super = y_train_super.astype(np.float32)
y_val_super = y_val_super.astype(np.float32)
y_test_super = y_test_super.astype(np.float32)


# # Defining Entropy and Metrics

# In[16]:


import tensorflow.keras.backend as K

def weighted_binary_crossentropy(weights):
    def loss(y_true, y_pred):
        weights_cast = K.cast(weights, y_pred.dtype)
        y_true = K.cast(y_true, y_pred.dtype)
        
        bce = K.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * weights_cast + (1 - y_true)
        weighted_bce = weight_vector * bce
        return K.mean(weighted_bce)
    return loss

def macro_f1(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    y_pred = K.round(y_pred)
    
    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def weighted_f1(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    y_pred = K.round(y_pred)
    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)
    support = K.sum(y_true, axis=0)
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    weighted_f1 = K.sum(f1 * support) / K.sum(support)
    weighted_f1 = tf.where(tf.math.is_nan(weighted_f1), 0.0, weighted_f1)
    
    return weighted_f1


# # Defining Models

# In[17]:


def create_cnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    return model


# In[18]:


# def create_resnet_model(input_shape, num_classes):
#     inputs = layers.Input(shape=input_shape)
#     x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
#     previous_filters = x.shape[-1]
#     for filters in [64, 128, 256]:
#         x_shortcut = x
#         strides = 1
#         if previous_filters != filters:
#             strides = 2

#         x = layers.Conv1D(filters, kernel_size=3, strides=strides, padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Conv1D(filters, kernel_size=3, padding='same')(x)
#         x = layers.BatchNormalization()(x)
        
#         if previous_filters != filters or strides != 1:
#             x_shortcut = layers.Conv1D(filters, kernel_size=1, strides=strides, padding='same')(x_shortcut)
#             x_shortcut = layers.BatchNormalization()(x_shortcut)
        
#         x = layers.Add()([x, x_shortcut])
#         x = layers.Activation('relu')(x)
#         previous_filters = filters
#     x = layers.GlobalAveragePooling1D()(x)
#     outputs = layers.Dense(num_classes, activation='sigmoid')(x)
#     model = models.Model(inputs, outputs)
#     return model


# In[19]:


def residual_block_1d(x, filters, kernel_size=3, strides=1, downsample=False):
    shortcut = x
    
    x = layers.Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def create_resnet_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    layers_filters = [64, 128, 256, 512]
    layers_blocks = [3, 4, 6, 3]

    for filters, num_blocks in zip(layers_filters, layers_blocks):
        for i in range(num_blocks):
            if i == 0 and filters != x.shape[-1]:
                x = residual_block_1d(x, filters, strides=2, downsample=True)
            else:
                x = residual_block_1d(x, filters)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model


# In[20]:


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def create_vit_model(input_shape, num_classes):
    patch_size = 10 
    num_patches = input_shape[0] // patch_size
    projection_dim = 64
    num_heads = 4
    transformer_layers = 8
    mlp_head_units = [256, 128]
    dropout_rate = 0.1

    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((num_patches, patch_size * input_shape[1]))(inputs)
    x = layers.Dense(units=projection_dim)(x)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    x = x + position_embedding(positions)
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate
        )(x1, x1)
        x2 = layers.Add()([attention_output, x])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=dropout_rate)
        x = layers.Add()([x3, x2])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# # Defining the training loop

# In[21]:


def train_model(
    model,
    X_train, y_train,
    X_val, y_val,
    class_weight,
    batch_size: int = 64,
    epochs: int = 25,
    loss_override=None,
    metrics_override=None,
    optimizer_override=None,
):
    optimizer = optimizer_override or tf.keras.optimizers.Adam()
    loss_fn = loss_override or 'binary_crossentropy'
    metrics = metrics_override or ['accuracy', macro_f1, weighted_f1]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history



# # Training and Evaluating Models without CL

# In[22]:


input_shape = X_train.shape[1:]
num_classes_super = y_train_super.shape[1]

cnn_super_model = create_cnn_model(input_shape, num_classes_super)
train_model(cnn_super_model, X_train, y_train_super, X_val, y_val_super, class_weight_super)


# In[23]:


resnet_super_model = create_resnet_model(input_shape, num_classes_super)
train_model(resnet_super_model, X_train, y_train_super, X_val, y_val_super, class_weight_super)


# In[24]:


vit_super_model = create_vit_model(input_shape, num_classes_super)
train_model(vit_super_model, X_train, y_train_super, X_val, y_val_super, class_weight_super)


# In[25]:


num_classes_sub = y_train_sub.shape[1]
cnn_sub_model = create_cnn_model(input_shape, num_classes_sub)
train_model(cnn_sub_model, X_train, y_train_sub, X_val, y_val_sub, class_weight_sub)


# In[26]:


resnet_sub_model = create_resnet_model(input_shape, num_classes_sub)
train_model(resnet_sub_model, X_train, y_train_sub, X_val, y_val_sub, class_weight_sub)


# In[27]:


vit_sub_model = create_vit_model(input_shape, num_classes_sub)
train_model(vit_sub_model, X_train, y_train_sub, X_val, y_val_sub, class_weight_sub)


# In[28]:


def evaluate_model(model, X_test, y_test, classes):
    y_pred = model.predict(X_test)
    y_pred_threshold = (y_pred >= 0.5).astype(int)
    report = classification_report(y_test, y_pred_threshold, target_names=classes, zero_division=0, output_dict=True)
    print(classification_report(y_test, y_pred_threshold, target_names=classes, zero_division=0))
    return report


# In[29]:


print("CNN Superdiagnostic Classification Report:")
cnn_super_report = evaluate_model(cnn_super_model, X_test, y_test_super, classes_super)

print("ResNet Superdiagnostic Classification Report:")
resnet_super_report = evaluate_model(resnet_super_model, X_test, y_test_super, classes_super)

print("ViT Superdiagnostic Classification Report:")
vit_super_report = evaluate_model(vit_super_model, X_test, y_test_super, classes_super)


# In[30]:


print("CNN Subdiagnostic Classification Report:")
cnn_sub_report = evaluate_model(cnn_sub_model, X_test, y_test_sub, classes_sub)

print("ResNet Subdiagnostic Classification Report:")
resnet_sub_report = evaluate_model(resnet_sub_model, X_test, y_test_sub, classes_sub)

print("ViT Subdiagnostic Classification Report:")
vit_sub_report = evaluate_model(vit_sub_model, X_test, y_test_sub, classes_sub)


# # Defining and Training on LwF

# In[31]:


cnn_soft_targets_super = cnn_super_model.predict(X_train)

def lwf_loss(y_true, y_pred, old_predictions, T=2):
    task_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dist_loss = tf.keras.losses.KLDivergence()(tf.nn.softmax(old_predictions / T),
                                               tf.nn.softmax(y_pred / T))
    total_loss = task_loss + dist_loss
    return total_loss

print("Working on CNN for LwF Now:")
cnn_model_lwf = create_cnn_model(input_shape, num_classes_sub)
cnn_model_lwf.compile(
    optimizer='adam',
    loss=lambda y_true, y_pred: lwf_loss(y_true, y_pred, old_predictions=cnn_soft_targets_super),
    metrics=[macro_f1, weighted_f1]
)
train_model(cnn_model_lwf, X_train, y_train_sub, X_val, y_val_sub, class_weight_sub)

print("Working on ResNet for LwF Now:")
resnet_soft_targets_super = resnet_super_model.predict(X_train)
resnet_model_lwf = create_resnet_model(input_shape, num_classes_sub)
resnet_model_lwf.compile(
    optimizer='adam',
    loss=lambda y_true, y_pred: lwf_loss(y_true, y_pred, old_predictions=resnet_soft_targets_super),
    metrics=[macro_f1, weighted_f1]
)
train_model(resnet_model_lwf, X_train, y_train_sub, X_val, y_val_sub, class_weight_sub)

print("Working on ViT for LwF Now:")
vit_soft_targets_super = vit_super_model.predict(X_train)
vit_model_lwf = create_vit_model(input_shape, num_classes_sub)
vit_model_lwf.compile(
    optimizer='adam',
    loss=lambda y_true, y_pred: lwf_loss(y_true, y_pred, old_predictions=vit_soft_targets_super),
    metrics=[macro_f1, weighted_f1]
)
train_model(vit_model_lwf, X_train, y_train_sub, X_val, y_val_sub, class_weight_sub)


# # Defining and Training on EwC

# In[32]:


class EWC:
    def __init__(self, model, X, y, batch_size=32, exclude_params=[]):
        self.model = model
        self.params = {}
        for p in model.trainable_variables:
            if id(p) not in exclude_params:
                self.params[id(p)] = p.numpy()
        self.fisher = self.compute_fisher(X, y, batch_size, exclude_params)

    def compute_fisher(self, X, y, batch_size, exclude_params):
        fisher = {}
        num_samples = X.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))

        for batch_idx in range(num_batches):
            X_batch = X[batch_idx*batch_size:(batch_idx+1)*batch_size]
            y_batch = y[batch_idx*batch_size:(batch_idx+1)*batch_size]
            with tf.GradientTape() as tape:
                preds = self.model(X_batch)
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_batch, preds))
            grads = tape.gradient(loss, self.model.trainable_variables)
            for p, g in zip(self.model.trainable_variables, grads):
                if g is not None and id(p) not in exclude_params:
                    param_id = id(p)
                    if param_id not in fisher:
                        fisher[param_id] = np.square(g.numpy())
                    else:
                        fisher[param_id] += np.square(g.numpy())
        for k in fisher.keys():
            fisher[k] /= num_batches
        return fisher

    def penalty(self, model):
        loss = 0
        for p in model.trainable_variables:
            param_id = id(p)
            if param_id in self.fisher:
                fisher = tf.convert_to_tensor(self.fisher[param_id])
                loss += tf.reduce_sum(fisher * tf.square(p - self.params[param_id]))
        return loss

def modify_model_for_subdiagnostic(base_model, num_classes_sub):
    inputs = base_model.input
    x = inputs
    for layer in base_model.layers[1:-1]:
        x = layer(x)
    outputs = layers.Dense(num_classes_sub, activation='sigmoid', name='output_sub')(x)
    new_model = models.Model(inputs=inputs, outputs=outputs)
    return new_model

lambda_ewc = 1000
cnn_sub_model = modify_model_for_subdiagnostic(cnn_super_model, num_classes_sub)
exclude_params_cnn = [id(w) for w in cnn_sub_model.layers[-1].trainable_weights]
ewc_cnn = EWC(cnn_super_model, X_train, y_train_super, exclude_params=exclude_params_cnn)

def ewc_loss_cnn(y_true, y_pred):
    task_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    ewc_penalty = ewc_cnn.penalty(cnn_sub_model)
    total_loss = task_loss + (lambda_ewc / 2) * ewc_penalty
    return total_loss

cnn_sub_model.compile(
    optimizer='adam',
    loss=ewc_loss_cnn,
    metrics=[macro_f1, weighted_f1]
)

# Use the trainer with loss_override so EWC is actually used
train_model(
    cnn_sub_model,
    X_train, y_train_sub,
    X_val, y_val_sub,
    class_weight_sub,
    loss_override=ewc_loss_cnn
)


def modify_model_for_subdiagnostic_resnet(base_model, num_classes_sub):
    x = base_model.layers[-2].output
    outputs = layers.Dense(num_classes_sub, activation='sigmoid', name='output_sub')(x)
    new_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return new_model

# ResNet sub head
num_classes_sub = y_train_sub.shape[1]
resnet_sub_model = modify_model_for_subdiagnostic_resnet(resnet_super_model, num_classes_sub)

# Exclude new head by ID (not names)
exclude_params_resnet = [id(w) for w in resnet_sub_model.layers[-1].trainable_weights]

# Instantiate EWC on the super model (snapshot on Task-0)
ewc_resnet = EWC(resnet_super_model, X_train, y_train_super, exclude_params=exclude_params_resnet)

def ewc_loss_resnet(y_true, y_pred):
    task_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    ewc_penalty = ewc_resnet.penalty(resnet_sub_model)
    return task_loss + (lambda_ewc / 2) * ewc_penalty

# Train with EWC loss
train_model(
    resnet_sub_model,
    X_train, y_train_sub,
    X_val, y_val_sub,
    class_weight_sub,
    loss_override=ewc_loss_resnet
)


def modify_model_for_subdiagnostic_vit(base_model, num_classes_sub):
    # Get the output of the layer before the last (excluding the superdiagnostic output layer)
    x = base_model.layers[-2].output
    # Add new output layer for subdiagnostic task
    outputs = tf.keras.layers.Dense(num_classes_sub, activation='sigmoid', name='output_sub')(x)
    # Create new model
    new_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    return new_model

# Modify ViT model for subdiagnostic task
# ViT sub head
num_classes_sub = y_train_sub.shape[1]
vit_sub_model = modify_model_for_subdiagnostic_vit(vit_super_model, num_classes_sub)

# Exclude new head by ID
exclude_params_vit = [id(w) for w in vit_sub_model.layers[-1].trainable_weights]

# Instantiate EWC on the super model
ewc_vit = EWC(vit_super_model, X_train, y_train_super, exclude_params=exclude_params_vit)

def ewc_loss_vit(y_true, y_pred):
    task_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    ewc_penalty = ewc_vit.penalty(vit_sub_model)
    return task_loss + (lambda_ewc / 2) * ewc_penalty

# Train with EWC loss
train_model(
    vit_sub_model,
    X_train, y_train_sub,
    X_val, y_val_sub,
    class_weight_sub,
    loss_override=ewc_loss_vit
)

class SI:
    def __init__(self, prev_model, damping_factor=0.1, exclude_params=[]):
        self.prev_params = {}
        self.omega = {}
        self.damping_factor = damping_factor
        self.exclude_params = exclude_params

        self.delta_params = {}

        # Store parameters from the previous model (superdiagnostic task)
        for var in prev_model.trainable_variables:
            if var.name not in self.exclude_params:
                self.prev_params[var.name] = var.numpy().copy()
                self.omega[var.name] = np.zeros_like(var.numpy())
                self.delta_params[var.name] = np.zeros_like(var.numpy())

    def accumulate_importance(self, model, grads):
        for var, grad in zip(model.trainable_variables, grads):
            if grad is not None and var.name in self.prev_params:
                if var.shape == self.prev_params[var.name].shape:
                    delta_theta = var.numpy() - self.prev_params[var.name]
                    self.delta_params[var.name] += delta_theta
                    # Update omega with absolute value to prevent negative importance
                    self.omega[var.name] += np.abs(grad.numpy() * delta_theta)
                else:
                    # Skip variables with mismatched shapes
                    pass

    def update_omega(self):
        # Normalize omega after training
        epsilon = 1e-8  # Small value to prevent division by zero
        for var_name in self.omega.keys():
            delta_param = self.delta_params[var_name]
            denom = np.square(delta_param) + self.damping_factor + epsilon
            self.omega[var_name] = np.divide(self.omega[var_name], denom)
            # Ensure omega is non-negative
            self.omega[var_name] = np.abs(self.omega[var_name])
            # Reset delta_params for the next task
            self.delta_params[var_name] = np.zeros_like(delta_param)

    def penalty(self, model):
        loss = 0
        for var in model.trainable_variables:
            if var.name in self.prev_params:
                prev_param = self.prev_params[var.name]
                if var.shape == prev_param.shape:
                    omega = tf.convert_to_tensor(self.omega[var.name], dtype=var.dtype)
                    prev_param = tf.convert_to_tensor(prev_param, dtype=var.dtype)
                    # Ensure omega is non-negative
                    loss += tf.reduce_sum(omega * tf.square(var - prev_param))
                else:
                    # Skip variables with mismatched shapes
                    pass
        return loss

num_classes_sub = y_train_sub.shape[1]
cnn_sub_model = modify_model_for_subdiagnostic(cnn_super_model, num_classes_sub)
exclude_params_cnn = [id(w) for w in cnn_sub_model.layers[-1].trainable_weights]
si_cnn = SI(cnn_super_model, exclude_params=exclude_params_cnn)

lambda_si = 1.0  # Adjust as needed
epochs = 25
batch_size = 64
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_macro_f1 = tf.keras.metrics.Mean(name='train_macro_f1')
train_loss = tf.keras.metrics.Mean(name='train_loss')
val_macro_f1 = tf.keras.metrics.Mean(name='val_macro_f1')
val_loss = tf.keras.metrics.Mean(name='val_loss')

for epoch in range(epochs):
    start_time = time.time()
    print(f'\nCNN Epoch {epoch+1}/{epochs}')
    train_macro_f1.reset_state()
    train_loss.reset_state()

    num_batches = len(X_train) // batch_size
    progress_bar = tqdm(range(num_batches), desc='Training', leave=False)

    for step in progress_bar:
        X_batch = X_train[step*batch_size:(step+1)*batch_size]
        y_batch = y_train_sub[step*batch_size:(step+1)*batch_size]

        with tf.GradientTape() as tape:
            preds = cnn_sub_model(X_batch, training=True)
            task_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_batch, preds))
            si_penalty = si_cnn.penalty(cnn_sub_model)
            total_loss = task_loss + (lambda_si / 2) * si_penalty

        grads = tape.gradient(total_loss, cnn_sub_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, cnn_sub_model.trainable_variables))
        si_cnn.accumulate_importance(cnn_sub_model, grads)

        batch_macro_f1 = macro_f1(y_batch, preds)
        train_macro_f1.update_state(batch_macro_f1)
        train_loss.update_state(total_loss)

        progress_bar.set_postfix({'loss': train_loss.result().numpy(), 'macro_f1': train_macro_f1.result().numpy()})

    epoch_time = time.time() - start_time

    val_macro_f1.reset_state()
    val_loss.reset_state()
    val_batches = len(X_val) // batch_size
    val_progress_bar = tqdm(range(val_batches), desc='Validation', leave=False)
    for step in val_progress_bar:
        X_batch = X_val[step*batch_size:(step+1)*batch_size]
        y_batch = y_val_sub[step*batch_size:(step+1)*batch_size]
        preds = cnn_sub_model(X_batch, training=False)
        task_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_batch, preds))
        total_loss = task_loss

        batch_macro_f1 = macro_f1(y_batch, preds)
        val_macro_f1.update_state(batch_macro_f1)
        val_loss.update_state(total_loss)

        val_progress_bar.set_postfix({'val_loss': val_loss.result().numpy(), 'val_macro_f1': val_macro_f1.result().numpy()})

    print(f'Epoch {epoch+1}/{epochs}, '
          f'Time: {epoch_time:.2f}s, '
          f'Loss: {train_loss.result():.4f}, '
          f'Macro F1: {train_macro_f1.result():.4f}, '
          f'Val Loss: {val_loss.result():.4f}, '
          f'Val Macro F1: {val_macro_f1.result():.4f}')

si_cnn.update_omega()

resnet_sub_model = modify_model_for_subdiagnostic_resnet(resnet_super_model, num_classes_sub)
exclude_params_resnet = [id(w) for w in resnet_sub_model.layers[-1].trainable_weights]
si_resnet = SI(resnet_super_model, exclude_params=exclude_params_resnet)

lambda_si = 1.0  # Adjust as needed
epochs = 25
batch_size = 64
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_macro_f1 = tf.keras.metrics.Mean(name='train_macro_f1')
train_loss = tf.keras.metrics.Mean(name='train_loss')
val_macro_f1 = tf.keras.metrics.Mean(name='val_macro_f1')
val_loss = tf.keras.metrics.Mean(name='val_loss')

for epoch in range(epochs):
    start_time = time.time()
    print(f'\nResNet Epoch {epoch+1}/{epochs}')
    train_macro_f1.reset_state()
    train_loss.reset_state()

    num_batches = len(X_train) // batch_size
    progress_bar = tqdm(range(num_batches), desc='Training', leave=False)

    for step in progress_bar:
        X_batch = X_train[step*batch_size:(step+1)*batch_size]
        y_batch = y_train_sub[step*batch_size:(step+1)*batch_size]

        with tf.GradientTape() as tape:
            preds = resnet_sub_model(X_batch, training=True)
            task_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_batch, preds))
            si_penalty = si_resnet.penalty(resnet_sub_model)
            total_loss = task_loss + (lambda_si / 2) * si_penalty

        grads = tape.gradient(total_loss, resnet_sub_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, resnet_sub_model.trainable_variables))
        si_resnet.accumulate_importance(resnet_sub_model, grads)

        batch_macro_f1 = macro_f1(y_batch, preds)
        train_macro_f1.update_state(batch_macro_f1)
        train_loss.update_state(total_loss)

        progress_bar.set_postfix({'loss': train_loss.result().numpy(), 'macro_f1': train_macro_f1.result().numpy()})

    epoch_time = time.time() - start_time

    # Validation
    val_macro_f1.reset_state()
    val_loss.reset_state()
    val_batches = len(X_val) // batch_size
    val_progress_bar = tqdm(range(val_batches), desc='Validation', leave=False)
    for step in val_progress_bar:
        X_batch = X_val[step*batch_size:(step+1)*batch_size]
        y_batch = y_val_sub[step*batch_size:(step+1)*batch_size]
        preds = resnet_sub_model(X_batch, training=False)
        task_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_batch, preds))
        total_loss = task_loss

        batch_macro_f1 = macro_f1(y_batch, preds)
        val_macro_f1.update_state(batch_macro_f1)
        val_loss.update_state(total_loss)

        val_progress_bar.set_postfix({'val_loss': val_loss.result().numpy(), 'val_macro_f1': val_macro_f1.result().numpy()})

    print(f'Epoch {epoch+1}/{epochs}, '
          f'Time: {epoch_time:.2f}s, '
          f'Loss: {train_loss.result():.4f}, '
          f'Macro F1: {train_macro_f1.result():.4f}, '
          f'Val Loss: {val_loss.result():.4f}, '
          f'Val Macro F1: {val_macro_f1.result():.4f}')

# After training, update omega
si_resnet.update_omega()


# In[44]:


vit_sub_model = modify_model_for_subdiagnostic_vit(vit_super_model, num_classes_sub)
exclude_params_vit = [id(w) for w in vit_sub_model.layers[-1].trainable_weights]
si_vit = SI(vit_super_model, exclude_params=exclude_params_vit)


# In[45]:


lambda_si = 1
epochs = 25
batch_size = 64
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_macro_f1 = tf.keras.metrics.Mean(name='train_macro_f1')
train_loss = tf.keras.metrics.Mean(name='train_loss')
val_macro_f1 = tf.keras.metrics.Mean(name='val_macro_f1')
val_loss = tf.keras.metrics.Mean(name='val_loss')

for epoch in range(epochs):
    start_time = time.time()
    print(f'\nViT Epoch {epoch+1}/{epochs}')
    train_macro_f1.reset_state()
    train_loss.reset_state()

    num_batches = len(X_train) // batch_size
    progress_bar = tqdm(range(num_batches), desc='Training', leave=False)

    for step in progress_bar:
        X_batch = X_train[step*batch_size:(step+1)*batch_size]
        y_batch = y_train_sub[step*batch_size:(step+1)*batch_size]

        with tf.GradientTape() as tape:
            preds = vit_sub_model(X_batch, training=True)
            task_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_batch, preds))
            si_penalty = si_vit.penalty(vit_sub_model)
            total_loss = task_loss + (lambda_si / 2) * si_penalty

        # Check for NaN in total_loss
        if tf.math.is_nan(total_loss):
            print(f"NaN detected in total_loss at epoch {epoch+1}, step {step+1}")
            break

        grads = tape.gradient(total_loss, vit_sub_model.trainable_variables)
        # Clip gradients to prevent exploding gradients
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]

        # Check for NaN in gradients
        if any([tf.reduce_any(tf.math.is_nan(g)) for g in grads if g is not None]):
            print(f"NaN detected in gradients at epoch {epoch+1}, step {step+1}")
            break

        optimizer.apply_gradients(zip(grads, vit_sub_model.trainable_variables))
        si_vit.accumulate_importance(vit_sub_model, grads)

        batch_macro_f1 = macro_f1(y_batch, preds)
        train_macro_f1.update_state(batch_macro_f1)
        train_loss.update_state(total_loss)

        progress_bar.set_postfix({'loss': train_loss.result().numpy(), 'macro_f1': train_macro_f1.result().numpy()})

    epoch_time = time.time() - start_time

    # Validation
    val_macro_f1.reset_state()
    val_loss.reset_state()
    val_batches = len(X_val) // batch_size
    val_progress_bar = tqdm(range(val_batches), desc='Validation', leave=False)
    for step in val_progress_bar:
        X_batch = X_val[step*batch_size:(step+1)*batch_size]
        y_batch = y_val_sub[step*batch_size:(step+1)*batch_size]
        preds = vit_sub_model(X_batch, training=False)
        task_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_batch, preds))
        total_loss = task_loss

        batch_macro_f1 = macro_f1(y_batch, preds)
        val_macro_f1.update_state(batch_macro_f1)
        val_loss.update_state(total_loss)

        val_progress_bar.set_postfix({'val_loss': val_loss.result().numpy(), 'val_macro_f1': val_macro_f1.result().numpy()})

    print(f'Epoch {epoch+1}/{epochs}, '
          f'Time: {epoch_time:.2f}s, '
          f'Loss: {train_loss.result():.4f}, '
          f'Macro F1: {train_macro_f1.result():.4f}, '
          f'Val Loss: {val_loss.result():.4f}, '
          f'Val Macro F1: {val_macro_f1.result():.4f}')

    # Check for NaN in training loss
    if tf.math.is_nan(train_loss.result()):
        print("NaN detected in training loss. Stopping training.")
        break

# After training, update omega
si_vit.update_omega()


# # Compiling and Publishing Results

# In[46]:


print("CNN EWC Subdiagnostic Classification Report:")
cnn_ewc_sub_report = evaluate_model(cnn_sub_model, X_test, y_test_sub, classes_sub)

print("CNN EWC Superdiagnostic Classification Report:")
cnn_ewc_super_report = evaluate_model(cnn_super_model, X_test, y_test_super, classes_super)

print("ResNet EWC Subdiagnostic Classification Report:")
resnet_ewc_sub_report = evaluate_model(resnet_sub_model, X_test, y_test_sub, classes_sub)

print("ResNet EWC Superdiagnostic Classification Report:")
resnet_ewc_super_report = evaluate_model(resnet_super_model, X_test, y_test_super, classes_super)

print("ViT EWC Subdiagnostic Classification Report:")
vit_ewc_sub_report = evaluate_model(vit_sub_model, X_test, y_test_sub, classes_sub)

print("ViT EWC Superdiagnostic Classification Report:")
vit_ewc_super_report = evaluate_model(vit_super_model, X_test, y_test_super, classes_super)


# In[47]:


print("CNN SI Subdiagnostic Classification Report:")
cnn_si_sub_report = evaluate_model(cnn_sub_model, X_test, y_test_sub, classes_sub)

print("CNN SI Superdiagnostic Classification Report:")
cnn_si_super_report = evaluate_model(cnn_super_model, X_test, y_test_super, classes_super)

print("ResNet SI Subdiagnostic Classification Report:")
resnet_si_sub_report = evaluate_model(resnet_sub_model, X_test, y_test_sub, classes_sub)

print("ResNet SI Superdiagnostic Classification Report:")
resnet_si_super_report = evaluate_model(resnet_super_model, X_test, y_test_super, classes_super)

print("ViT SI Subdiagnostic Classification Report:")
vit_si_sub_report = evaluate_model(vit_sub_model, X_test, y_test_sub, classes_sub)

print("ViT SI Superdiagnostic Classification Report:")
vit_si_super_report = evaluate_model(vit_super_model, X_test, y_test_super, classes_super)


# In[48]:


def get_macro_f1(report_dict):
    return report_dict['macro avg']['f1-score']

results = {
    'Model': [],
    'Task': [],
    'Macro F1-score': []
}

results['Model'].extend(['CNN', 'ResNet', 'ViT'])
results['Task'].extend(['Superdiagnostic'] * 3)
results['Macro F1-score'].extend([
    get_macro_f1(cnn_super_report),
    get_macro_f1(resnet_super_report),
    get_macro_f1(vit_super_report)
])

results['Model'].extend(['CNN', 'ResNet', 'ViT'])
results['Task'].extend(['Subdiagnostic'] * 3)
results['Macro F1-score'].extend([
    get_macro_f1(cnn_sub_report),
    get_macro_f1(resnet_sub_report),
    get_macro_f1(vit_sub_report)
])

results['Model'].extend(['CNN', 'ResNet', 'ViT'])
results['Task'].extend(['EWC Subdiagnostic'] * 3)
results['Macro F1-score'].extend([
    get_macro_f1(cnn_ewc_sub_report),
    get_macro_f1(resnet_ewc_sub_report),
    get_macro_f1(vit_ewc_sub_report)
])

results['Model'].extend(['CNN', 'ResNet', 'ViT'])
results['Task'].extend(['EWC Superdiagnostic'] * 3)
results['Macro F1-score'].extend([
    get_macro_f1(cnn_ewc_super_report),
    get_macro_f1(resnet_ewc_super_report),
    get_macro_f1(vit_ewc_super_report)
])

results['Model'].extend(['CNN', 'ResNet', 'ViT'])
results['Task'].extend(['SI Subdiagnostic'] * 3)
results['Macro F1-score'].extend([
    get_macro_f1(cnn_si_sub_report),
    get_macro_f1(resnet_si_sub_report),
    get_macro_f1(vit_si_sub_report)
])

results['Model'].extend(['CNN', 'ResNet', 'ViT'])
results['Task'].extend(['SI Superdiagnostic'] * 3)
results['Macro F1-score'].extend([
    get_macro_f1(cnn_si_super_report),
    get_macro_f1(resnet_si_super_report),
    get_macro_f1(vit_si_super_report)
])

results_df = pd.DataFrame(results)
print("\nSummary of Classification Performance:")
print(results_df)

