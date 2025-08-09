# Differentiable Synthesizer for Text-to-Audio Generation

This project implements a text-to-audio generation system using a differentiable synthesizer guided by the CLAP (Contrastive Language-Audio Pretraining) model. The goal is to generate audio that matches a user's text description by iteratively optimizing the parameters of a synthesizer.

## Core Concept

We treat audio generation as an optimization problem. Instead of directly generating waveforms, we search for the optimal set of synthesizer parameters (`θ`) that produce a sound whose audio embedding (`e_audio`) closely matches the text embedding (`e_text`) of the user's prompt. This allows us to leverage the descriptive power of natural language to control a complex audio synthesis engine.

## The Synthesizer

The `synthesizer.py` file contains a `DifferentiableSynth` class. The current version is a feature-rich subtractive synthesizer that includes:
- **Dual Oscillator Source:** A blend between a pure sine wave oscillator and filtered white noise.
- **Pitch Envelope:** A dedicated envelope to control the pitch of the oscillator over time.
- **Time-Varying Resonant Filter:** A resonant low-pass filter whose cutoff frequency is modulated by its own dedicated ADSR envelope. This is the core component for creating dynamic effects like "wah-wah" sweeps.
- **Amplitude Envelope:** A full ADSR (Attack, Decay, Sustain, Release) envelope to shape the overall volume of the sound.
- **High-Performance Filtering:** The time-varying filter is implemented using a high-performance, vectorized "block processing" approach with `torchaudio.functional.lfilter`, which eliminates Python loops for maximum speed.

### Optimizable Parameters (`theta`)

The synthesizer exposes a comprehensive set of raw parameters for optimization. These are the "knobs" that the CLAP model can turn to find the desired sound.

- **Oscillator:**
    - `noise_mix_raw`: Controls the blend between the sine wave and noise.
- **Pitch Envelope:**
    - `start_freq_raw`: The initial pitch of the oscillator.
    - `end_freq_raw`: The final pitch of the oscillator.
    - `pitch_decay_raw`: The time it takes for the pitch to glide from start to end.
- **Amplitude ADSR Envelope:**
    - `amp_attack_raw`: The time it takes for the sound to reach maximum volume.
    - `amp_decay_raw`: The time it takes to decay to the sustain level.
    - `amp_sustain_raw`: The volume level held while the note is active.
    - `amp_release_raw`: The time it takes for the sound to fade out after the note ends.
- **Filter & Filter ADSR Envelope:**
    - `filter_cutoff_raw`: The base cutoff frequency of the low-pass filter.
    - `filter_q_raw`: The resonance or "peak" of the filter.
    - `filter_env_amount_raw`: How much the filter envelope affects the cutoff frequency (can be positive or negative).
    - `filt_env_attack_raw`: The attack time of the filter's envelope.
    - `filt_env_decay_raw`: The decay time of the filter's envelope.
    - `filt_env_sustain_raw`: The sustain level of the filter's envelope.
    - `filt_env_release_raw`: The release time of the filter's envelope.

## The Flow: From Text to Sound

The end-to-end process remains the same, but now operates on the much richer synthesizer described above.

### 1. User Input (The Prompt)
The process begins with a user providing a descriptive text prompt.
- **Example:** `"A funky 'wah-wah' effect on a synthesizer."`

### 2. CLAP (Text) → Target Embedding
The system feeds the text prompt into the pre-trained CLAP model to get a "target embedding." This embedding is a single vector that mathematically represents the semantic meaning of the desired sound.

### 3. The Optimization Loop
The system iteratively refines the synthesizer parameters to minimize the distance between the generated audio's embedding and the target text embedding. For each step in the loop:
- **a. Synthesize:** Generate an audio clip using the current parameters, `θ_i`.
- **b. Analyze:** Create a log-Mel spectrogram of the audio.
- **c. Embed:** Feed the spectrogram into the CLAP model's audio encoder to get `e_synth`.
- **d. Calculate Loss:** Measure the cosine similarity between `e_synth` and `e_target`.
- **e. Backpropagate:** Calculate the gradients of the loss with respect to `θ_i`.
- **f. Optimize:** Apply the gradients to update `θ_i` using the AdamW optimizer.
- **g. Repeat.**

### 4. Final Output
Once the optimization loop is finished, the final, optimized parameter set `θ_final` is used to render the audio to a `.wav` file.

---

## How to Run

The `optimizer.py` script is the main entry point. It contains the full optimization loop and configuration.

You can run the entire process by executing this file:
```bash
python diff_synth/optimizer.py
```
This will start the optimization and, after a minute or two, save the resulting audio to `optimized_sound.wav`. You can change the `TEXT_PROMPT` variable and the initial `theta_raw` dictionary in `optimizer.py` to generate different sounds.

---

## Dependencies

You will need Python 3.8+ and a recent version of PyTorch.

### Core Libraries:
- `torch`
- `torchaudio`
- `transformers`
- `dasp-pytorch`
- `librosa`
- `soundfile`
- `scipy`
- `numpy`

You can install them via pip:
```bash
pip install torch torchaudio transformers dasp-pytorch librosa soundfile scipy numpy
```
