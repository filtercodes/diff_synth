import os
import logging

# Suppress the specific warning from torch.distributed.elastic
log = logging.getLogger("torch.distributed.elastic.multiprocessing.redirects")
log.setLevel(logging.ERROR)

import torch
import torch.optim as optim
import torchaudio
import librosa
import soundfile as sf
import time
import warnings

from transformers import AutoProcessor, ClapModel
from synthesizer import DifferentiableSynth, scale


# Suppress the specific UserWarning from PyTorch/Transformers
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated.*"
)

# --- Configuration ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SYNTH_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SAMPLE_RATE = 48000
# --- Prompts for Creative Exploration ---
TEXT_PROMPT = "A sharp, metallic, electronic hi-hat with a very short decay."
# TEXT_PROMPT = "A deep, resonant, analog synthesizer bass note with a slow, opening filter."
# TEXT_PROMPT = "The sound of a resonant filter sweep on a synthesizer pad."
# TEXT_PROMPT = "A classic 8-bit video game laser zap sound."
# TEXT_PROMPT = "A soft, airy, synthesizer pad with a gentle, swelling attack."
# TEXT_PROMPT = "A funky 'wah-wah' effect on a synthesizer."
# TEXT_PROMPT = "An aggressive, distorted, industrial synth stab."
# TEXT_PROMPT = "A bright, shimmering, glass-like bell tone."

# Optimization Hyperparameters
LEARNING_RATE = 0.05
NUM_ITERATIONS = 200
PRINT_INTERVAL = 20

def main():
    """
    Main function to run the text-to-audio optimization loop.
    """
    print(f"--- Using CLAP model on device: {DEVICE} ---")
    print(f"--- Using Synthesizer and Spectrogram on device: {SYNTH_DEVICE} ---")

    # 1. Load CLAP Model and Text Processor
    print("Loading CLAP model and processor...")
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(DEVICE)
    text_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap_model.eval()
    print("CLAP model loaded.")

    # 2. Create Differentiable Audio Preprocessing Pipeline on CPU
    # This pipeline must run on the CPU because the STFT operation inside
    # MelSpectrogram produces complex numbers, which are not supported by the MPS backend.
    audio_pipeline = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=64, f_min=20, f_max=20000,
        ),
        torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80),
    ).to("cpu")
    print("Differentiable audio processing pipeline created on CPU.")

    # 3. Instantiate the Differentiable Synthesizer with a longer duration
    synth = DifferentiableSynth(sample_rate=SAMPLE_RATE, duration_s=4.0)
    print("Differentiable synthesizer instantiated.")

    # 4. Define Initial RAW Synthesizer Parameters (Theta) on the CPU
    # All parameters are trainable for maximum creative freedom.
    theta_raw = {
        # Start with a PURE TONE, not noise. Large negative value -> sigmoid(x) -> 0
        'noise_mix_raw': torch.tensor([-5.0], requires_grad=True), 
        
        # Start with a low-ish frequency
        'start_freq_raw': torch.tensor([-2.0], requires_grad=True), # -> ~260 Hz (C4)
        'end_freq_raw': torch.tensor([-2.0], requires_grad=True),
        'pitch_decay_raw': torch.tensor([0.0], requires_grad=True),
        
        # Start with a standard "pluck" or "key" shape
        'amp_attack_raw': torch.tensor([-5.0], requires_grad=True), # Fast attack
        'amp_decay_raw': torch.tensor([-1.0], requires_grad=True), # Medium decay
        'amp_sustain_raw': torch.tensor([-1.0], requires_grad=True),# Low-ish sustain
        'amp_release_raw': torch.tensor([0.0], requires_grad=True),
        
        # Start with the filter mostly closed, ready to be opened by the envelope
        'filter_cutoff_raw': torch.tensor([-2.0], requires_grad=True), # Low cutoff
        'filter_q_raw': torch.tensor([1.0], requires_grad=True),       # Medium resonance
        'filter_env_amount_raw': torch.tensor([0.0], requires_grad=True), # Let it learn this
        
        # A standard filter envelope shape
        'filt_env_attack_raw': torch.tensor([-4.0], requires_grad=True),
        'filt_env_decay_raw': torch.tensor([0.0], requires_grad=True),
        'filt_env_sustain_raw': torch.tensor([-1.0], requires_grad=True),
        'filt_env_release_raw': torch.tensor([0.0], requires_grad=True),
    }
    theta_raw = {k: v.to("cpu") for k, v in theta_raw.items()}
    
    print(f"Initial raw parameters (theta_raw): { {k: v.item() for k, v in theta_raw.items()} }")

    # 5. Set up the Optimizer
    # AdamW is generally more stable and effective for complex creative tasks.
    optimizer = optim.AdamW(list(theta_raw.values()), lr=0.01, weight_decay=1e-4)
    print(f"Optimizer: AdamW with learning rate 0.01")

    # 6. Get the Target Text Embedding
    print(f"\n--- Preparing target embedding for prompt: '{TEXT_PROMPT}' ---")
    target_start_time = time.time()
    with torch.no_grad():
        text_inputs = text_processor(text=[TEXT_PROMPT], return_tensors="pt", padding=True).to(DEVICE)
        target_embedding = clap_model.get_text_features(**text_inputs)
    target_end_time = time.time()
    print(f"Target embedding generated in {target_end_time - target_start_time:.2f} seconds.")

    # --- 7. The Optimization Loop ---
    print(f"\n--- Starting optimization for {NUM_ITERATIONS} iterations ---")
    loop_start_time = time.time()

    for i in range(NUM_ITERATIONS):
        iter_start_time = time.time()
        optimizer.zero_grad()
        
        # --- Benchmarking: Synthesizer Forward Pass ---
        synth_start_time = time.time()
        synthesized_audio = synth.forward(theta_raw, SYNTH_DEVICE)
        synth_end_time = time.time()
        
        # --- Calculate Spectrogram (on CPU, then move to GPU) ---
        synthesized_audio_cpu = synthesized_audio.cpu()
        log_mel_spec_cpu = audio_pipeline(synthesized_audio_cpu)
        log_mel_spec_cpu = log_mel_spec_cpu.transpose(-1, -2)
        log_mel_spec = log_mel_spec_cpu.to(DEVICE)
        
        # --- Benchmarking: CLAP Loss Calculation ---
        loss_start_time = time.time()
        clap_input_spec = log_mel_spec.unsqueeze(1)
        synth_embedding = clap_model.get_audio_features(input_features=clap_input_spec)
        loss = -torch.nn.functional.cosine_similarity(target_embedding, synth_embedding).mean()
        loss_end_time = time.time()
        
        # --- Benchmarking: Backward Pass ---
        backward_start_time = time.time()
        loss.backward()
        backward_end_time = time.time()

        # --- Gradient Clipping ---
        torch.nn.utils.clip_grad_norm_(theta_raw.values(), max_norm=1.0)

        # --- Benchmarking: Optimizer Step ---
        optimizer_start_time = time.time()
        optimizer.step()
        optimizer_end_time = time.time()

        if (i + 1) % PRINT_INTERVAL == 0:
            with torch.no_grad():
                start_f = scale(torch.sigmoid(theta_raw['start_freq_raw']), 0, 1, 20., 8000.)
                a_decay = scale(torch.sigmoid(theta_raw['amp_decay_raw']), 0, 1, 0.01, 2.0)
                f_cutoff = scale(torch.sigmoid(theta_raw['filter_cutoff_raw']), 0, 1, 100.0, 12000.0)
                f_env = scale(torch.tanh(theta_raw['filter_env_amount_raw']), -1, 1, -8000.0, 8000.0)
            
            iter_end_time = time.time()
            print(f"Iter {i+1}/{NUM_ITERATIONS} | Loss: {loss.item():.4f} | Freq: {start_f.item():.0f}Hz | ADecay: {a_decay.item():.2f}s | FiltCut: {f_cutoff.item():.0f}Hz | FiltEnv: {f_env.item():.0f}Hz")
            print(f"  Timings (s): Total: {iter_end_time - iter_start_time:.2f} | Synth: {synth_end_time - synth_start_time:.2f} | Loss: {loss_end_time - loss_start_time:.2f} | Backward: {backward_end_time - backward_start_time:.2f} | Optim: {optimizer_end_time - optimizer_start_time:.2f}")

    loop_end_time = time.time()
    print(f"--- Optimization finished in {loop_end_time - loop_start_time:.2f} seconds ---")

    # 8. Final Results
    print("\n--- Final Results ---")
    print(f"Final Loss: {loss.item():.4f}")
    with torch.no_grad():
        # Retrieve and scale all final parameters for printing
        start_f = scale(torch.sigmoid(theta_raw['start_freq_raw']), 0, 1, 20., 8000.)
        end_f = scale(torch.sigmoid(theta_raw['end_freq_raw']), 0, 1, 20., 8000.)
        p_decay = scale(torch.sigmoid(theta_raw['pitch_decay_raw']), 0, 1, 0.01, 2.0)
        a_decay = scale(torch.sigmoid(theta_raw['amp_decay_raw']), 0, 1, 0.01, 2.0)
        mix = torch.sigmoid(theta_raw['noise_mix_raw'])
        
        f_cutoff = scale(torch.sigmoid(theta_raw['filter_cutoff_raw']), 0, 1, 100.0, 12000.0)
        f_q = scale(torch.sigmoid(theta_raw['filter_q_raw']), 0, 1, 0.707, 10.0)
        f_env_amt = scale(torch.tanh(theta_raw['filter_env_amount_raw']), -1, 1, -8000.0, 8000.0)
        
        a_att = scale(torch.sigmoid(theta_raw['amp_attack_raw']), 0, 1, 0.001, 1.0)
        a_sus = torch.sigmoid(theta_raw['amp_sustain_raw'])
        a_rel = scale(torch.sigmoid(theta_raw['amp_release_raw']), 0, 1, 0.01, 2.0)

        f_att = scale(torch.sigmoid(theta_raw['filt_env_attack_raw']), 0, 1, 0.001, 1.0)
        f_dec = scale(torch.sigmoid(theta_raw['filt_env_decay_raw']), 0, 1, 0.01, 1.0)
        f_sus = torch.sigmoid(theta_raw['filt_env_sustain_raw'])
        f_rel = scale(torch.sigmoid(theta_raw['filt_env_release_raw']), 0, 1, 0.01, 1.0)

        print(f"  Osc: Freq {start_f.item():.0f}->{end_f.item():.0f}Hz | PDecay: {p_decay.item():.2f}s | Mix: {mix.item():.2f}")
        print(f"  Amp Env: A:{a_att.item():.3f}s D:{a_decay.item():.3f}s S:{a_sus.item():.2f} R:{a_rel.item():.3f}s")
        print(f"  Filt: Cutoff {f_cutoff.item():.0f}Hz | Q: {f_q.item():.2f} | EnvAmt: {f_env_amt.item():.0f}Hz")
        print(f"  Filt Env: A:{f_att.item():.3f}s D:{f_dec.item():.3f}s S:{f_sus.item():.2f} R:{f_rel.item():.3f}s")


    # 9. Generate and Save Final Audio
    print("\nGenerating final audio with optimized parameters...")
    with torch.no_grad():
        final_audio = synth.forward(theta_raw, SYNTH_DEVICE)

    output_filename = "optimized_sound.wav"
    audio_to_save = final_audio.detach().cpu().squeeze(0).numpy()
    sf.write(output_filename, audio_to_save, SAMPLE_RATE)
    print(f"Successfully saved final audio to '{output_filename}'")

if __name__ == '__main__':
    main()
