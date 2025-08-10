# 🧠 Unpaired MRI ↔ CT Image Translation for Brain Tumour Segmentation

A deep learning framework using **CycleGAN** for unpaired **MRI ↔ CT image translation**, aimed at bridging modality gaps in medical imaging for better **brain tumour segmentation**.  
This approach applies **Generative AI (GenAI)** concepts to medical imaging, enabling cross-modality synthesis without paired datasets, improving segmentation in low-data or low-resource clinical settings.

---

## 🤖 Introduction to Generative AI

**Generative AI (GenAI)** refers to a branch of Artificial Intelligence focused on creating new content — such as images, text, audio, or video — that resembles real-world data.  
Unlike traditional AI models that classify or predict from existing inputs, generative models **learn data distributions** and can produce synthetic, yet realistic, outputs.

In medical imaging, Generative AI can:
- **Generate synthetic scans** to augment limited datasets.
- **Translate between modalities** (e.g., MRI ↔ CT).
- **Improve cross-domain performance** in downstream tasks like segmentation or diagnosis.

Popular Generative AI approaches include:
- **GANs (Generative Adversarial Networks)**
- **VAEs (Variational Autoencoders)**
- **Diffusion Models**

---

## 🧠 What are GANs?

A **Generative Adversarial Network (GAN)**, introduced by Goodfellow et al. (2014), is a framework with two neural networks competing in a zero-sum game:

- **Generator (G)**: Produces synthetic samples from random noise or a source domain.
- **Discriminator (D)**: Tries to distinguish between real and synthetic samples.

Training Process:
1. The generator tries to produce outputs indistinguishable from real data.
2. The discriminator evaluates whether the sample is real or fake.
3. Both models improve iteratively — the generator gets better at “fooling” the discriminator, and the discriminator gets better at spotting fakes.

This **adversarial training** produces highly realistic synthetic data.

---

## 🔄 From GAN to CycleGAN

While standard GANs require paired data (e.g., an MRI image and its exact CT counterpart), **CycleGAN** (Zhu et al., 2017) enables **unpaired image-to-image translation**.

**Key innovation:**  
CycleGAN introduces **cycle-consistency** — if you translate an image from Domain A → Domain B → back to Domain A, you should retrieve the original image.

---

## 🏗 CycleGAN Architecture

### Components:
1. **Two Generators:**
   - \( G_{MRI \to CT} \): Converts MRI images into CT-like images.
   - \( G_{CT \to MRI} \): Converts CT images into MRI-like images.

2. **Two Discriminators:**
   - \( D_{CT} \): Distinguishes between real CT and generated CT images.
   - \( D_{MRI} \): Distinguishes between real MRI and generated MRI images.

### Loss Functions:
- **Adversarial Loss**: Encourages generated images to be indistinguishable from real ones in the target domain.
- **Cycle-Consistency Loss**: Ensures A→B→A returns the original input.
- **Identity Loss**: Ensures that translating an image already in the target domain doesn’t alter it unnecessarily.
- **SSIM Loss (Structural Similarity)**: Preserves anatomical structures.
- **Feature Adaptation Loss**: Aligns high-level features between domains using pretrained networks.

![CycleGAN Architecture](assets/cyclegan_architecture.png)

---

## 📜 Problem Statement

**Context:**
- MRI offers better soft tissue contrast — ideal for tumour localisation.
- CT is faster, cheaper, and better for bone imaging.

**Challenge:**  
Segmentation models trained on MRI often fail on CT, and vice versa, due to **domain shift** — statistical differences in pixel intensity, contrast, and textures between modalities.

**Real-world limitation:**  
Acquiring paired MRI and CT scans for the same patients is rare and expensive.  
Thus, we need an **unpaired translation** approach to bridge the modality gap.

**Goal:**  
Translate between MRI and CT while preserving structural and pathological information, enabling improved cross-modality segmentation.

---

## 📂 Dataset

- **Source**: Jordan University Hospital (JUH)
- **Patients**: 20
- **Images**: 178 axial 2D slices (90 MRI, 88 CT), resized to 256×256
- **Labels**: Tumour masks annotated by radiologists
- **Setting**: Unpaired MRI and CT datasets (different patients for each modality)

**Example Images with Masks:**
![Dataset with Masks](assets/dataset_masks.png)

---

## 🛠 Preprocessing

Preprocessing ensures model stability and better generalisation.

1. Convert to **grayscale**
2. Resize to **256×256**
3. Normalise pixel values to [0, 1]
4. Add channel dimension
5. Augmentation:
   - Rotations (±30°)
   - Horizontal & vertical flips
   - Contrast changes

**Pipeline:**
![Preprocessing Pipeline](assets/preprocessing_pipeline.png)

---

## 📋 Methodology

### 1️⃣ CycleGAN Translation
- Unpaired training between MRI and CT.
- Feature adaptation via pretrained ResNet/VGG features.
- Additional SSIM loss for structural fidelity.

### 2️⃣ U-Net Segmentation
- Evaluates impact of synthetic data on tumour segmentation.
- Metrics: Dice Coefficient & IoU.

### 3️⃣ Evaluation
- Compare baseline segmentation vs segmentation trained with synthetic modality augmentation.

---

## 🖼 Sample Translations

MRI → CT (top) and CT → MRI (bottom):

![Sample Translations](assets/sample_translations.png)

---

## 📊 Segmentation Results

Segmentation metrics visualised below:

![Segmentation Metrics](assets/segmentation_metrics.png)

---

## 📚 Key Takeaways

- **Generative AI in medical imaging** can help bridge modality gaps.
- **CycleGAN** enables unpaired MRI↔CT translation, preserving critical structures.
- **Feature adaptation + SSIM loss** improves perceptual quality (SSIM ↑ vs baseline GANs).
- Downstream segmentation performance benefits from synthetic modality augmentation.

---

## 📧 Contact

**Author:** Suchit Pathak  
📩 **Email:** suchitpathak0807@gmail.com  
💻 **GitHub:** [github.com/Suchit0807](https://github.com/Suchit0807)  
🌐 **Portfolio:** [suchit0807.github.io/suchit-portfolio](https://suchit0807.github.io/suchit-portfolio/)  
🌐 **LinkedIn:** [linkedin.com/in/suchitpathak](https://linkedin.com/in/suchitpathak)
---

**⭐ If you found this work useful, please consider starring the repo!**

