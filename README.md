    `Breast Cancer Segmentation` is a project made by ~ Devansh Madake
<hr>

**NOTE: USE COMMAND `git lfs clone https://github.com/creativekids11/Breast-Cancer-Mammography-Segmentation.git` for cloning this repo, due to large dataset files**

In this breast cancer segmentation model is trained on data of CBIS-DDSM, MIAS, IN-Breast.
A lot thanks for the data providers to give the data for free

# Summary of used type of: ACAAtrousResUNet

**ACAAtrousResUNet** is a hybrid segmentation head that reuses a pretrained **ResNet34 encoder** (via `segmentation_models_pytorch.Unet`) and replaces the usual decoder with:

- an **ASPP (atrous spatial pyramid pooling)** module on the encoder bottleneck to capture multiscale context, and then
- a stack of **UpACA** decoder blocks that perform upsampling + **ACA attention** (channel + spatial attention that refines encoder skip features using the decoder “gate”), followed by regular convolutional fusion.

So the net combines the representational power of a ResNet encoder, multiscale context (ASPP), and attention-guided skip fusion (ACA) in a U-Net–like upsampling path.

---

## Components (what they are & why)

### DoubleConv
Two conv layers (3×3) each followed by `BatchNorm` + `ReLU`. Standard U‑Net style block for local feature extraction and refinement. Optional small `Dropout2d` appended when `dropout=True`.

**Effect:** learns stronger local features, reduces checkerboard artifacts vs single conv.

### Down
`MaxPool2d(2)` then `DoubleConv`: encoder downsampling block. Produces progressively smaller spatial maps with more channels.

### Up
A conventional upsampling block: bilinear upsample (or `ConvTranspose` if `bilinear=False`) → concat with encoder skip → `DoubleConv`.

Used in the plain `UNet` implementation.

### ACAModule (the core of ACA)
This is the attention block that refines the encoder skip-map using the decoder features (the “gate”):

- It concatenates `skip` and `gate` along channel dim.
- `ca` (channel attention) path: reduces channels → `ReLU` → project back → `Sigmoid`. Produces a per-channel attention vector that modulates `skip`.
- `spatial` path: a 3×3 conv (padding=1) that reduces channels → produces a single-channel spatial mask → `Sigmoid`. Gives location-specific weighting.
- Result: `refined = skip * ca * sa + skip` (residual-style refinement).
- `fuse` : `1×1` Conv + `BatchNorm` + `ReLU` to mix channels after refinement.

**Effect:** the decoder feature `gate` guides which channels and spatial positions of the encoder `skip` are most relevant before concatenation. This reduces irrelevant information passed through skip connections.

### UpACA
Decoder upsample block that:

- Upsamples decoder tensor,
- If not already created, lazily constructs an `ACAModule` and a `DoubleConv` matching the real skip and gate channel sizes (determined at first forward; this avoids having to know encoder channel counts at init time),
- Applies ACA to refine the skip, concatenates refined skip + upsampled decoder, then applies `DoubleConv`.

**Important:** It lazily creates submodules on first call (so the dummy forward initialization elsewhere is important to register parameters with optimizer).

### ASPP (Atrous Spatial Pyramid Pooling)
- Multiple parallel dilated convs with dilation rates in `rates=(1,6,12,18)` by default.
- Each conv outputs `out_ch` channels → they are concatenated → `BN` + `ReLU` → `1×1` projection back to `out_ch`.

**Purpose:** capture multiscale context and enlarge receptive field without reducing spatial resolution — useful for small/varied tumour sizes.

### ACAAtrousUNet
A pure-from-scratch U-Net variant that stacks the `Down`/`ASPP`/`UpACA` components. Useful as a baseline when you don't want pretrained encoders.

### ACAAtrousResUNet (the model described)
- `self.encoder = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=in_ch, classes=out_ch)`  
  Uses `segmentation_models_pytorch.Unet` primarily to access a `ResNet34` encoder (`self.encoder.encoder`), whose extracted feature maps will be used.  
  **Note:** specifying `in_channels=1` with `encoder_weights="imagenet"` commonly produces a warning because ImageNet weights expect 3 channels; `smp` adjusts the first conv or reinitializes it but pretrained benefit may be reduced for single-channel input.

- `encoder_channels = self.encoder.encoder.out_channels` — collects channel counts for each encoder stage (used to size `ASPP` and `UpACA` constructors).

- `self.aspp = ASPP(in_ch=encoder_channels[-1], out_ch=encoder_channels[-2])` — runs `ASPP` on the deepest encoder feature (bottleneck) and projects to a channel count matching the previous encoder stage for decoder compatibility.

- `self.up_aca1..4` — a chain of `UpACA` blocks that progressively upsample from the `ASPP` output and refine encoder skips `e4, e3, e2, e1`.

- `self.outc` — `1×1` conv to produce final logits; final output interpolated to input resolution.

---

## Forward pass (step-by-step)

1. `feats = self.encoder.encoder(x)` — get the list of intermediate encoder features from ResNet34. Typical order in `smp` is `[stage0, stage1, stage2, stage3, stage4, stage5]` with `stage5` being deepest (bottleneck).
2. Extract `e1..e4` and `bottleneck` (the code handles variable-length by indexing from end if needed).
3. `d5 = self.aspp(bottleneck)` — apply multi-rate context on bottleneck.
4. `d4 = self.up_aca1(d5, e4)` — upsample and refine with ACA using encoder feature `e4`.
5. Continue: `d3 = up_aca2(d4, e3)`, `d2 = up_aca3(d3, e2)`, `d1 = up_aca4(d2, e1)`.
6. `logits = self.outc(d1)` and return interpolated logits to input size.

---

## Why this architecture?

- **ResNet encoder** provides strong, pretrained features (edges → semantics).
- **ASPP** gives large receptive field and multiscale context — important when tumor sizes vary widely.
- **ACA** improves skip connection usefulness by removing irrelevant encoder features (channel + spatial gating guided by decoder state), which helps reduce false positive propagation from encoder to decoder.
- Combining these produces a decoder that is both context-aware and selective about skip information.

<hr>

Though this model may seem complex it offer a fast inference on codes

# Data Preproccessing
* The data is pre-proccessed with convectional CLAHE and Median filter.
* Just for experimental purposes the texture(glcm maps) overlay has been done also, which aids in giving a minute squeeze of dice scores.

# Testing Part
* For testing the simple and fast cv2 platform is made with guide printed in terminal on execution
* There is also a adaptive CLAHE feature which helps in giving the model more confidence on confusing images
* The inference is done on 1024 x 1024 and 512 x 512 sizes of image. Such that the model being trained on 512 sized images tends to find more finer and small minute parts of cancer!
