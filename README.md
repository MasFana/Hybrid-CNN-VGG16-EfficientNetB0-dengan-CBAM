# Hybrid-CNN-VGG16-EfficientNetB0-dengan-CBAM

## **Arsitektur Hybrid CNN: VGG16 + EfficientNetB0 dengan CBAM**
**Berdasarkan Paper**

Rani R, Bharany S, Elkamchouchi DH, Ur Rehman A, Singh R, Hussen S. VGG-EffAttnNet: Hybrid Deep Learning Model for Automated Chili Plant Disease Classification Using VGG16 and EfficientNetB0 With Attention Mechanism. Food Sci Nutr. 2025 Jul 24;13(7):e70653. doi: 10.1002/fsn3.70653. PMID: 40708782; PMCID: PMC12288623.

**Dataset** : Chili Plant Leaf Disease and Growth Stage Dataset from Bangladesh
 https://data.mendeley.com/datasets/w9mr3vf56s/1
### **Diagram Arsitektur**

```
INPUT (224, 224, 3)
     ↓
  ┌─────────────────────────────────────────────────┐
  │              PREPROCESSING LAYERS               │
  │ • Rescaling (0-255 → 0-1)                       │
  │ • Normalization (Custom)                        │
  │ • Rescaling_1 (Additional)                      │
  └─────────────────────────────────────────────────┘
     ↓
  ┌─────────────────┐    ┌─────────────────────────────┐
  │   BRANCH 1      │    │       BRANCH 2              │
  │   VGG16 PATH    │    │   EfficientNetB0 + CBAM     │
  │                 │    │                             │
  │ Conv2D (64)     │    │ Stem Layers                 │
  │ Conv2D (64)     │    │ (Conv2D + BN + Activation)  │
  │ MaxPooling2D    │    │     ↓                       │
  │ Conv2D (128)    │    │ MBConv Blocks 1-7           │
  │ Conv2D (128)    │    │ with SE Attention           │
  │ MaxPooling2D    │    │     ↓                       │
  │ Conv2D (256)    │    │ Top Conv (1280 channels)    │
  │ Conv2D (256)    │    │     ↓                       │
  │ Conv2D (256)    │    │ CBAM Block:                 │
  │ MaxPooling2D    │    │   - GlobalAvgPool +         │
  │ Conv2D (512)    │    │     GlobalMaxPool           │
  │ Conv2D (512)    │    │   - Dense(80) + Dense(1280) │
  │ Conv2D (512)    │    │   - Add + Activation        │
  │ MaxPooling2D    │    │   - Multiply (Channel Att.) │
  │ Conv2D (512)    │    │   - Lambda (Spatial Avg/Max)│
  │ Conv2D (512)    │    │   - Concatenate + Conv2D    │
  │ Conv2D (512)    │    │   - Multiply (Spatial Att.) │
  │ MaxPooling2D    │    │     ↓                       │
  │ Flatten()       │    │ GlobalAveragePooling2D      │
  │     ↓           │    │     ↓                       │
  │ Dense(256)      │    │ Dense(256)                  │
  │     ↓           │    │     ↓                       │
  │ Dropout(0.5)    │    │ Dropout(0.5)                │
  └─────────────────┘    └─────────────────────────────┘
               ↘                   ↙
            ┌─────────────────────────┐
            │    FEATURE FUSION       │
            │                         │
            │ Concatenate() → (512)   │
            │       ↓                 │
            │ Dense(512) + Dropout    │
            │       ↓                 │
            │ Dense(256) + Dropout    │
            │       ↓                 │
            │ Output Layer (6 classes)│
            │ (Softmax Activation)    │
            └─────────────────────────┘
```

### **Detail Implementasi Arsitektur Hybrid**

#### **Input Processing & Preprocessing**
- **Dimensi Input**: 224×224×3 (RGB)
- **Preprocessing Layers**:
  - `Rescaling`: Normalisasi pixel values (0-255 → 0-1)
  - `Normalization`: Layer normalisasi custom dengan 7 parameter
  - `Rescaling_1`: Layer rescaling tambahan

#### **Dual-Branch Feature Extraction**

**Branch 1 - VGG16 Pathway:**
```
VGG16_Input →
[Conv2D(64) → Conv2D(64) → MaxPooling2D] →
[Conv2D(128) → Conv2D(128) → MaxPooling2D] →
[Conv2D(256) → Conv2D(256) → Conv2D(256) → MaxPooling2D] →
[Conv2D(512) → Conv2D(512) → Conv2D(512) → MaxPooling2D] →
[Conv2D(512) → Conv2D(512) → Conv2D(512) → MaxPooling2D] →
Flatten() → Dense(256, ReLU) → Dropout(0.5) → VGG_Features
```

**Branch 2 - EfficientNetB0 + CBAM Pathway:**
```
EfficientNet_Input →
Stem (Conv2D+BN+Activation) →
MBConv Blocks 1-7 dengan SE Attention →
Top Conv (1280 channels) →
CBAM_Attention_Block →
GlobalAveragePooling2D →
Dense(256, ReLU) → Dropout(0.5) → EffNet_Features
```

#### **CBAM Block Implementation Detail**
```
Input Features (7,7,1280)
     ↓
Channel Attention Path:
├─ GlobalAveragePooling2D → (1280)
├─ GlobalMaxPooling2D → (1280)
├─ Dense(80) → Dense(1280) (kedua path)
├─ Add → Activation (sigmoid) → Channel Weights
└─ Multiply → Channel-Refined Features

Spatial Attention Path:
├─ Lambda (Avg Pool per channel) → (7,7,1)
├─ Lambda (Max Pool per channel) → (7,7,1)
├─ Concatenate → (7,7,2)
├─ Conv2D(7×7, filters=1) → (7,7,1)
├─ Activation (sigmoid) → Spatial Weights
└─ Multiply → Final Refined Features
```

#### **Feature Fusion & Classification Head**
```
[VGG_Features (256)] ⊕ [EffNet_Features (256)] → (512)
     ↓
Dense(512, ReLU) → Dropout(0.5)
     ↓
Dense(256, ReLU) → Dropout(0.5)
     ↓
Dense(6, Softmax) → Output
```

### **Spesifikasi Teknis Berdasarkan Model Summary**

#### **Parameter Distribution**
- **Total Parameters**: 26,322,924 (100.41 MB)
- **Trainable Parameters**: 21,431,097 (81.75 MB) - **81.4%**
- **Non-trainable Parameters**: 4,891,827 (18.66 MB) - **18.6%**

#### **Dimensional Flow Aktual**
```
Input: (224, 224, 3)
    ↓
VGG16 Path:
  (224,224,3) → ... → (7,7,512) → Flatten → (25088) → Dense → (256)

EfficientNetB0 + CBAM Path:
  (224,224,3) → ... → (7,7,1280) → CBAM → (7,7,1280) → GAP → (1280) → Dense → (256)
    ↓
Concatenate: (512) → Dense → (512) → Dense → (256) → Output → (6)
```

#### **Key Implementation Details**

1. **EfficientNetB0 Base**:
   - Menggunakan arsitektur EfficientNetB0 lengkap dengan MBConv blocks
   - Memiliki mekanisme Squeeze-and-Excitation (SE) attention internal
   - Output: (7,7,1280) sebelum CBAM

2. **CBAM Implementation**:
   - **Channel Attention**: Menggunakan kedua GlobalAvgPool dan GlobalMaxPool
   - **Spatial Attention**: Menggunakan average dan max pooling per channel
   - **Kernel Size**: Conv2D 7×7 untuk spatial attention

3. **VGG16 Implementation**:
   - Arsitektur VGG16 lengkap dengan 5 blok konvolusi
   - Output final: (7,7,512) sebelum flattening

4. **Fusion Strategy**:
   - Kedua branch menghasilkan features 256-dimensional
   - Concatenation menghasilkan 512-dimensional feature vector
   - Multiple dense layers dengan dropout untuk final classification

### **Keunggulan Arsitektur yang Diimplementasi**

1. **Comprehensive Feature Extraction**:
   - VGG16 menangkap hierarchical features yang stabil
   - EfficientNetB0 memberikan efficient feature extraction dengan scaling
   - CBAM meningkatkan focus pada region dan channel penting

2. **Advanced Attention Mechanism**:
   - SE attention dalam EfficientNet + CBAM external attention
   - Dual-path attention (channel + spatial)
   - Complementary attention mechanisms

3. **Optimized Transfer Learning**:
   - 81.4% parameter trainable untuk fine-tuning optimal
   - 18.6% parameter frozen untuk menjaga pre-trained knowledge

4. **Robust Regularization**:
   - Multiple dropout layers (4 dropout layers total)
   - Feature fusion dengan dimensionality reduction

### **Performance Optimization**

- **Memory Efficiency**: ~100MB model size untuk kemampuan hybrid
- **Feature Complementarity**: VGG16's stability + EfficientNet's efficiency
- **Attention Refinement**: CBAM meningkatkan feature quality
- **Multi-scale Processing**: Different receptive fields dari kedua architecture

Arsitektur ini menggabungkan kekuatan dari tiga approach modern: VGG16 untuk feature extraction yang terbukti, EfficientNet untuk efficiency, dan CBAM untuk adaptive feature refinement, menghasilkan model yang powerful untuk tugas classification.
