# PER (Percentile) Score

## Source

Unsupervised Spectrum Anomaly Detection Method for Unauthorized Bands\
Yu Tian et al., 2022

------------------------------------------------------------------------

## 1. What is PER Score?

The **PER (Percentile) Score** is an anomaly detection metric proposed
for unsupervised wireless spectrum anomaly detection using an
Autoencoder.

Instead of averaging reconstruction errors (like MSE or MAE), PER
focuses on the **high-percentile reconstruction errors** to capture
distribution shifts caused by anomalies.

It is designed to detect two VAE reconstruction effects:

-   **Background Noise Enhancement (BNE)**
-   **Anomaly Signal Disappearance (ASD)**

These effects create heavy tails in the reconstruction error
distribution, which PER explicitly measures using percentile statistics.

------------------------------------------------------------------------

## 2. Field of Application

PER is used in:

-   Wireless spectrum anomaly detection
-   Unauthorized frequency band monitoring
-   Cognitive radio systems
-   Unsupervised deep learning-based signal detection

It operates on time-frequency spectrogram data generated from wireless
signals.

------------------------------------------------------------------------

## 3. Mathematical Representation

The PER score is defined as:

PER(X, X̂) = β · q_ξ ( f_γ(\|X − X̂\|) ∘ r\_{α,γ}(X) ) + q_η ( f_γ(\|X −
X̂\|) ∘ (1 − r\_{α,γ}(X)) )

Where:

-   X: original spectrogram
-   X̂: reconstructed spectrogram
-   \|X − X̂\|: pixel-level reconstruction error
-   ∘: element-wise (Hadamard) product
-   β: weight for background region
-   ξ, η: percentile parameters
-   γ: pooling kernel size
-   α: threshold parameter

------------------------------------------------------------------------

## 4. Component Definitions

### Percentile Operator

q_ξ(X): value at the ξ-th percentile of the cumulative distribution of
X.

### Min-Pooling Operator

f_γ(X): min-pooling with kernel size γ, used to detect noise floor
variations.

### Background Mask

r\_{α,γ}(X) = ω_α ∘ h_γ(X)

-   h_γ(X): max-pooling
-   ω_α(X) = u(α − x\_{i,j})
-   u(x) = 1 if x ≥ 0, else 0

The mask separates:

-   Background region
-   Signal region

------------------------------------------------------------------------

## 5. Interpretation

PER = β × (high-percentile background error)\
+ (high-percentile signal error)

Typical parameter values in the paper:

-   γ = 3
-   β = 2
-   α = 0.05
-   ξ = 90
-   η = 99

------------------------------------------------------------------------

## 6. Why PER Outperforms MSE

MSE averages reconstruction errors across all pixels, which can dilute
short-duration or narrow-band anomalies.

PER instead:

-   Focuses on tail behavior of error distribution
-   Emphasizes worst-reconstructed regions
-   Is robust to anomaly dilution
-   Improves recall under fixed false alarm rate

------------------------------------------------------------------------

## 7. Key Takeaway

PER is a percentile-based reconstruction metric that enhances
unsupervised spectrum anomaly detection by capturing distribution shifts
caused by anomaly-induced reconstruction effects (BNE and ASD).
