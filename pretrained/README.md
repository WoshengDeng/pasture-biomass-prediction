### Pretrained Models

- **DINOv3 ViT-Huge**: Used as the primary vision backbone for end-to-end biomass regression.
- **SigLIP (so400m-patch14-384)**: Used as a frozen vision-language encoder to extract semantic embeddings.
  - Source: `google-siglip-so400m-patch14-384`
  - Role: Provides complementary semantic features for GBDT-based prediction branch.