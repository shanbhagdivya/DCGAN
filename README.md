# **DCGAN on CelebA Dataset (Kaggle)**
This project trains a **Deep Convolutional Generative Adversarial Network (DCGAN)** on the **CelebA dataset** to generate realistic human face images.

---

## **Dataset Preprocessing Steps**
### **1. Dataset Location**
- The CelebA dataset is taken from Kaggle

### **2. Preprocessing Steps**
- Resize images to **64x64 pixels**.
- Apply **center crop** to ensure facial features remain centered.
- Convert images to **tensors**.
- Normalize pixel values to **[-1, 1]** (for stability in training).

```python
transform = transforms.Compose([
  transforms.Resize(image_size),
  transforms.CenterCrop(image_size),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

### **3. Use Only 10% of the Dataset**
- To speed up training, we use only 10% of the dataset.

```python
total_size = len(dataset)
subset_size = int(0.1 * total_size)
indices = random.sample(range(total_size), subset_size)
subset = Subset(dataset, indices)
dataloader = data.DataLoader(subset, batch_size=batch_size, shuffle=True)
```

## Training the Model ##
### Running the Training Script ###
To train the DCGAN, simply run the Kaggle notebook. The key steps are:
- Train the Discriminator to differentiate real vs. fake images.
- Train the Generator to produce realistic images.
- Use Binary Cross-Entropy Loss (BCELoss).
- Optimize using Adam optimizer.

```python
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
```

## Testing the Model ##
### 1. Generate Fake Images ###
Once the model is trained, we can generate new face images using random noise:

```python
with torch.no_grad():
    fake_images = netG(torch.randn(5, nz, 1, 1, device=device)).cpu()
```

### 2. Compare Real vs. Generated Images
Instead of saving images, we display 5 pairs of real and generated images using matplotlib:

```python
fig, axes = plt.subplots(5, 2, figsize=(8, 10))
for i in range(5):
    axes[i, 0].imshow(np.transpose(real_images[i].numpy(), (1, 2, 0)) * 0.5 + 0.5)
    axes[i, 0].axis("off")
    axes[i, 0].set_title("Original")
    
    axes[i, 1].imshow(np.transpose(fake_images[i].numpy(), (1, 2, 0)) * 0.5 + 0.5)
    axes[i, 1].axis("off")
    axes[i, 1].set_title("Generated")

plt.show()
```

## Expected Output ##
- Discriminator Loss (D Loss): Should gradually decrease.
- Generator Loss (G Loss): Should increase initially, then stabilize.
- Generated Images:
    a. At early epochs: Noisy, unrealistic faces.
    b. After training: More structured, human-like faces.
