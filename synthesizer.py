import torch
import torch.nn as nn
import math
from phase import parallel_cumsum
import dasp_pytorch.signal as dsp_signal
import torchaudio.functional as F

def scale(val, src_min, src_max, dst_min, dst_max):
    """Scales a tensor from a source range to a destination range."""
    return (val - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min

def generate_adsr_envelope(attack_s, decay_s, sustain_level, release_s, num_samples, sample_rate, device='cpu'):
    """
    Generates a full ADSR envelope in a fully differentiable manner.
    This version uses torch.where for robust segment combination.
    """
    # Convert times from seconds to samples, maintaining tensor properties for autograd
    attack_samples = attack_s * sample_rate
    decay_samples = decay_s * sample_rate
    release_samples = release_s * sample_rate

    # Calculate segment boundary points
    attack_end = attack_samples
    decay_end = attack_samples + decay_samples
    
    # Ensure sustain is not negative if total duration is shorter than A+D+R
    sustain_end = torch.tensor(num_samples, device=device) - release_samples
    sustain_end = torch.max(decay_end, sustain_end) # Sustain can't end before decay

    # Create time steps as a column vector for broadcasting
    t = torch.arange(num_samples, device=device, dtype=torch.float32)

    # --- Calculate ramps for each phase ---

    # Attack: ramp from 0 to 1
    attack_ramp = t / torch.clamp(attack_samples, min=1e-5)
    
    # Decay: ramp from 1 down to sustain_level
    decay_ramp = 1.0 - ((t - attack_end) / torch.clamp(decay_samples, min=1e-5)) * (1.0 - sustain_level)
    
    # Release: ramp from sustain_level down to 0
    release_ramp = sustain_level * (1.0 - (t - sustain_end) / torch.clamp(release_samples, min=1e-5))

    # --- Combine segments using torch.where ---
    # This is like a nested if/else statement for tensors.
    # It builds the envelope from the inside out.
    
    # Start with the sustain level.
    # THIS IS THE CORRECTED LINE:
    envelope = sustain_level.expand_as(t)
    
    # Overlay the decay and release ramps
    envelope = torch.where(t < decay_end, decay_ramp, envelope)
    envelope = torch.where(t >= sustain_end, release_ramp, envelope)

    # Overlay the attack ramp last
    envelope = torch.where(t < attack_end, attack_ramp, envelope)

    # Final clamp to ensure values are in the valid [0, 1] range
    envelope = torch.clamp(envelope, 0.0, 1.0)
    
    return envelope

class DifferentiableSynth(nn.Module):
    """
    A differentiable synthesizer with a correct, time-varying resonant filter.
    """
    def __init__(self, sample_rate=48000, duration_s=2.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration_s = duration_s
        self.num_samples = int(sample_rate * duration_s)

    def forward(self, theta_raw: dict[str, torch.Tensor], synthesis_device: str) -> torch.Tensor:
        # --- 1. Parameter Scaling (Identical to before) ---
        theta = {k: v.to(synthesis_device) for k, v in theta_raw.items()}
        epsilon = 1e-6
        noise_mix = torch.sigmoid(theta['noise_mix_raw'])
        start_freq = scale(torch.sigmoid(theta['start_freq_raw']), 0, 1, 20.0, 8000.0)
        end_freq = scale(torch.sigmoid(theta['end_freq_raw']), 0, 1, 20.0, 8000.0)
        pitch_decay_time = scale(torch.sigmoid(theta['pitch_decay_raw']), 0, 1, 0.01, 2.0)
        amp_attack = scale(torch.sigmoid(theta['amp_attack_raw']), 0, 1, 0.001, 1.0)
        amp_decay = scale(torch.sigmoid(theta['amp_decay_raw']), 0, 1, 0.01, 2.0)
        amp_sustain = torch.sigmoid(theta['amp_sustain_raw'])
        amp_release = scale(torch.sigmoid(theta['amp_release_raw']), 0, 1, 0.01, 2.0)
        filter_cutoff_base = scale(torch.sigmoid(theta['filter_cutoff_raw']), 0, 1, 100.0, 12000.0)
        filter_q = scale(torch.sigmoid(theta['filter_q_raw']), 0, 1, 0.707, 10.0)
        filter_env_amount = scale(torch.tanh(theta['filter_env_amount_raw']), -1, 1, -8000.0, 8000.0)
        filt_env_attack = scale(torch.sigmoid(theta['filt_env_attack_raw']), 0, 1, 0.001, 1.0)
        filt_env_decay = scale(torch.sigmoid(theta['filt_env_decay_raw']), 0, 1, 0.01, 1.0)
        filt_env_sustain = torch.sigmoid(theta['filt_env_sustain_raw'])
        filt_env_release = scale(torch.sigmoid(theta['filt_env_release_raw']), 0, 1, 0.01, 1.0)

        # --- 2. Control Signal Generation (Identical to before) ---
        t = torch.linspace(0., self.duration_s, self.num_samples, device=synthesis_device)
        pitch_envelope_curve = torch.exp(-t / (pitch_decay_time + epsilon))
        time_varying_frequency = start_freq + (end_freq - start_freq) * (1 - pitch_envelope_curve)
        filter_envelope = generate_adsr_envelope(
            filt_env_attack, filt_env_decay, filt_env_sustain, filt_env_release,
            self.num_samples, self.sample_rate, device=synthesis_device
        )
        time_varying_cutoff = filter_cutoff_base + filter_envelope * filter_env_amount
        time_varying_cutoff = torch.clamp(time_varying_cutoff, 20.0, self.sample_rate / 2.1)

        # --- 3. Source Signal Generation (Identical to before) ---
        noise_signal = torch.rand(self.num_samples, device=synthesis_device) * 2 - 1
        angular_frequency = 2 * torch.pi * time_varying_frequency / self.sample_rate
        instantaneous_phase = parallel_cumsum(angular_frequency)
        sine_wave = torch.sin(instantaneous_phase)
        source_signal = (1.0 - noise_mix) * sine_wave + noise_mix * noise_signal
        
        # --- 4. TIME-VARYING Filter using Fast Block Processing ---
        BLOCK_SIZE = 1024 # Larger block size for better performance
        original_num_samples = self.num_samples

        remainder = original_num_samples % BLOCK_SIZE
        if remainder != 0:
            pad_amount = BLOCK_SIZE - remainder
            source_signal = torch.nn.functional.pad(source_signal, (0, pad_amount))
            time_varying_cutoff = torch.nn.functional.pad(time_varying_cutoff, (0, pad_amount))
        else:
            pad_amount = 0

        padded_num_samples = original_num_samples + pad_amount
        num_blocks = padded_num_samples // BLOCK_SIZE

        source_blocks = source_signal.view(num_blocks, BLOCK_SIZE)
        cutoff_blocks = time_varying_cutoff.view(num_blocks, BLOCK_SIZE)
        
        # --- Step 5: Vectorized Coefficient Calculation & Normalization ---
        block_cutoffs = cutoff_blocks.mean(dim=1)
        block_q = filter_q.expand(num_blocks)
        block_gain_db = torch.tensor([0.0], device=synthesis_device).expand(num_blocks)

        b_block, a_block = dsp_signal.biquad(
            block_gain_db, block_cutoffs, block_q, self.sample_rate, "low_pass"
        )

        epsilon = 1e-8
        a0_block = a_block[:, 0].unsqueeze(1) + epsilon
        b_norm = b_block / a0_block
        a_norm = a_block / a0_block

        # --- Step 6: Apply Filter with a SINGLE VECTORIZED lfilter Call ---
        # This one function call replaces the entire slow for loop.
        # CORRECTED to use the right keyword arguments: waveform, a_coeffs, b_coeffs
        filtered_blocks = F.lfilter(
            waveform=source_blocks,
            a_coeffs=a_norm,
            b_coeffs=b_norm
        )

        # --- Step 7: Reshape back to a single audio stream and trim padding ---
        filtered_signal_padded = filtered_blocks.view(-1)
        filtered_signal = filtered_signal_padded[:original_num_samples]

        # --- 5. Apply Final Amplitude Envelope ---
        amp_envelope = generate_adsr_envelope(
            amp_attack, amp_decay, amp_sustain, amp_release,
            self.num_samples, self.sample_rate, device=synthesis_device
        )
        audio_out = filtered_signal * amp_envelope

        return audio_out.unsqueeze(0)