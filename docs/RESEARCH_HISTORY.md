# Generative-model research history and evidence map

This repository records a progression from adversarial objectives to
conditional translation, diffusion, progressive generation, perceptual
super-resolution, and controllable face rendering. The projects are connected
by a recurring systems concern: generation must balance realism, coverage,
conditioning, and stability.

## Research arc

### 1. Adversarial objective and optimization stability

The DCGAN, WGAN, and WGAN-GP implementations explore binary adversarial loss,
weight-clipped Wasserstein critics, and gradient-penalized critics. The
conditional variant injects digit labels into both generator and critic.

**Available evidence:** model/training source and 42 archived MNIST grids.

**Current validation:** shared support diagnostics and conditional
cross-checkpoint consistency.

**Missing for reproduction:** exact environments, checkpoints, loss logs, and
controlled runs using aligned seeds/checkpoint budgets.

### 2. Conditional and unpaired image translation

Pix2Pix combines a conditional discriminator with an L1 reconstruction term for
paired edge-to-photo translation. CycleGAN and coupled-GAN code explore
unpaired or cross-domain translation.

**Available evidence:** 24 Pix2Pix condition/output pairs and one CycleGAN
showcase.

**Current validation:** Pix2Pix edge adherence, background spill, and explicit
failure cases.

**Missing:** ground-truth Pix2Pix targets, CycleGAN test sets/checkpoints, and
portable dataset paths.

### 3. Diffusion and progressive generation

The diffusion prototype implements a linear beta schedule, time-conditioned
UNet noise prediction, and iterative reverse sampling on 64×64 CelebA-HQ.
Progressive GAN code grows resolution through fade-in blocks.

**Available evidence:** diffusion/training source, 221 diffusion samples, and
progressive-GAN source.

**Current validation:** file-level diffusion inventory only.

**Missing:** CelebA-HQ reference split, checkpoints, training curves, sampling
seeds, and progressive-GAN results.

### 4. Perceptual reconstruction

The super-resolution branches combine residual generators, discriminators, and
VGG-based perceptual loss. They preserve the architectural intent but include
legacy Windows-style dataset paths and no paired result set.

**Status:** historical code; not currently reproducible.

### 5. Controllable generative representation

The StyleGAN/differentiable-renderer archive contains separate pose and
illumination control trajectories. These are the strongest evidence that the
repository moved beyond unconditional sampling toward interpretable control.

**Current validation:** 44 trajectory audits covering roughness, path
linearity, and edge continuity. Illumination is measurably smoother than pose,
which identifies a concrete axis-specific reliability gap.

## Senior-level synthesis

A single attractive sample cannot establish a reliable generative system. The
appropriate evaluation depends on the product contract:

- unconditional generation needs coverage and fidelity;
- conditional generation needs controllability as well as sample quality;
- translation needs adherence and artifact containment;
- controllable rendering needs smoothness, identity stability, and physical
  plausibility.

The new benchmark makes that contract explicit and preserves failures instead
of selecting only favorable outputs.

## Evidence taxonomy

| Label | Meaning |
|---|---|
| Artifact-audited | versioned outputs are measured deterministically, without claiming retraining |
| Inventory only | artifact quantity/format is checked; quality is not established |
| Historical showcase | one or a few visual results exist without a complete evaluation protocol |
| Historical code | architecture/source exists, but required data, checkpoint, or output evidence is absent |

No modern quality claim should be added without the exact reference dataset,
split, checkpoint, inference seeds, and evaluation implementation.
