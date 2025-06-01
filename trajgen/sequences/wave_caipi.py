"""
Defines the WaveCAIPISequence class.
"""
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from trajgen.generators import generate_wave_caipi_trajectory
from trajgen.sequence_base import MRISequence
from trajgen.trajectory import COMMON_NUCLEI_GAMMA_HZ_PER_T

__all__ = ['WaveCAIPISequence']


class WaveCAIPISequence(MRISequence):
    """
    Represents a Wave-CAIPI (Controlled Aliasing in Parallel Imaging) sequence.

    This implementation focuses on the 2D k-space trajectory aspects of Wave-EPI,
    where sinusoidal gradients are applied along the phase-encode direction,
    modulated along the readout.
    """

    def __init__(self,
                 # Common MRISequence parameters
                 name: str,
                 fov_mm: Union[float, Tuple[float, float]], # For the base EPI
                 resolution_mm: Union[float, Tuple[float, float]], # For the base EPI
                 dt_seconds: float,
                 # Wave-CAIPI specific parameters (mostly from generator)
                 num_echoes: int, # Total phase-encode lines before undersampling
                 points_per_echo: int, # Readout points
                 wave_amplitude_mm: float,
                 wave_frequency_cycles_per_fov_readout: float,
                 wave_phase_offset_rad: float = 0.0,
                 epi_type: str = 'flyback',
                 phase_encode_direction: str = 'y',
                 undersampling_factor_pe: float = 1.0,
                 # Other Trajectory/MRISequence parameters
                 gamma_Hz_per_T: float = COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'],
                 dead_time_start_seconds: float = 0.0,
                 dead_time_end_seconds: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initializes a WaveCAIPISequence object.

        Args:
            name (str): Name of the sequence.
            fov_mm (Union[float, Tuple[float, float]]): FOV for the base EPI.
            resolution_mm (Union[float, Tuple[float, float]]): Resolution for the base EPI.
            dt_seconds (float): Dwell time in seconds.
            num_echoes (int): Total number of phase-encode lines before undersampling.
            points_per_echo (int): Number of readout points per echo.
            wave_amplitude_mm (float): Amplitude of FOV shift by wave gradient (mm).
            wave_frequency_cycles_per_fov_readout (float): Wave frequency in cycles/FOV_readout.
            wave_phase_offset_rad (float): Phase offset for the wave.
            epi_type (str): 'flyback' or 'gradient_recalled'.
            phase_encode_direction (str): 'y' (kx readout, ky phase) or 'x' (ky readout, kx phase).
            undersampling_factor_pe (float): Undersampling factor in phase-encode direction.
            gamma_Hz_per_T (float): Gyromagnetic ratio.
            dead_time_start_seconds (float): Dead time at sequence start.
            dead_time_end_seconds (float): Dead time at sequence end.
            metadata (Optional[Dict[str, Any]]): Additional metadata.
        """
        self.num_echoes = num_echoes
        self.points_per_echo = points_per_echo
        self.wave_amplitude_mm = wave_amplitude_mm
        self.wave_frequency_cycles_per_fov_readout = wave_frequency_cycles_per_fov_readout
        self.wave_phase_offset_rad = wave_phase_offset_rad
        self.epi_type = epi_type
        self.phase_encode_direction = phase_encode_direction
        self.undersampling_factor_pe = undersampling_factor_pe

        sequence_specific_params = {
            'num_echoes': num_echoes,
            'points_per_echo': points_per_echo,
            'wave_amplitude_mm': wave_amplitude_mm,
            'wave_frequency_cycles_per_fov_readout': wave_frequency_cycles_per_fov_readout,
            'wave_phase_offset_rad': wave_phase_offset_rad,
            'epi_type': epi_type,
            'phase_encode_direction': phase_encode_direction,
            'undersampling_factor_pe': undersampling_factor_pe,
        }

        # Wave-CAIPI (as implemented by this generator) is 2D
        num_dimensions = 2

        super().__init__(
            name=name,
            fov_mm=fov_mm,
            resolution_mm=resolution_mm,
            num_dimensions=num_dimensions,
            dt_seconds=dt_seconds,
            gamma_Hz_per_T=gamma_Hz_per_T,
            sequence_specific_params=sequence_specific_params,
            dead_time_start_seconds=dead_time_start_seconds,
            dead_time_end_seconds=dead_time_end_seconds,
            metadata=metadata
        )

    def _generate_kspace_points(self) -> np.ndarray:
        """
        Generates k-space points for the Wave-CAIPI sequence.
        """
        kspace_points_D_N = generate_wave_caipi_trajectory(
            fov_mm=self.fov_mm,
            resolution_mm=self.resolution_mm,
            num_echoes=self.num_echoes,
            points_per_echo=self.points_per_echo,
            wave_amplitude_mm=self.wave_amplitude_mm,
            wave_frequency_cycles_per_fov_readout=self.wave_frequency_cycles_per_fov_readout,
            wave_phase_offset_rad=self.wave_phase_offset_rad,
            epi_type=self.epi_type,
            phase_encode_direction=self.phase_encode_direction,
            undersampling_factor_pe=self.undersampling_factor_pe,
            gamma_Hz_per_T=self.metadata.get('gamma_Hz_per_T', COMMON_NUCLEI_GAMMA_HZ_PER_T['1H'])
        )
        return kspace_points_D_N

    def check_gradient_limits(self, system_limits: Dict[str, Any]) -> bool:
        max_grad_limit_Tm_per_m = system_limits.get('max_grad_Tm_per_m')
        max_slew_limit_Tm_per_s = system_limits.get('max_slew_Tm_per_s_per_m')
        actual_max_grad_Tm = self.get_max_grad_Tm()
        actual_max_slew_Tm_per_s = self.get_max_slew_Tm_per_s()
        grad_ok, slew_ok = True, True
        print_prefix = f"Gradient Limit Check for '{self.name}':"

        if actual_max_grad_Tm is None: grad_ok = False; print(f"{print_prefix} Could not determine actual max gradient.")
        elif max_grad_limit_Tm_per_m is not None:
            grad_ok = actual_max_grad_Tm <= max_grad_limit_Tm_per_m
            print(f"{print_prefix} Max Gradient: {actual_max_grad_Tm:.4f} T/m. Limit: {max_grad_limit_Tm_per_m:.4f} T/m. Status: {'OK' if grad_ok else 'EXCEEDED'}")
        else: print(f"{print_prefix} Max Gradient: {actual_max_grad_Tm:.4f} T/m. No limit provided.")

        if actual_max_slew_Tm_per_s is None: slew_ok = False; print(f"{print_prefix} Could not determine actual max slew rate.")
        elif max_slew_limit_Tm_per_s is not None:
            slew_ok = actual_max_slew_Tm_per_s <= max_slew_limit_Tm_per_s
            print(f"{print_prefix} Max Slew Rate: {actual_max_slew_Tm_per_s:.2f} T/m/s. Limit: {max_slew_limit_Tm_per_s:.2f} T/m/s. Status: {'OK' if slew_ok else 'EXCEEDED'}")
        else: print(f"{print_prefix} Max Slew Rate: {actual_max_slew_Tm_per_s:.2f} T/m/s. No limit provided.")
        return grad_ok and slew_ok

    def assess_kspace_coverage(self) -> str:
        return (f"Wave-CAIPI sequence '{self.name}' based on {self.epi_type} EPI. "
                f"Covers k-space with {self.num_echoes // self.undersampling_factor_pe:.0f} acquired echoes "
                f"of {self.points_per_echo} points each. "
                "The wave modulation distributes k-space information along the phase-encode direction, "
                "creating a distinct pattern for CAIPI unfolding.")

    def estimate_off_resonance_sensitivity(self) -> str:
        return (f"Wave-CAIPI ('{self.name}'), being EPI-based, is sensitive to off-resonance effects, "
                "which can cause blurring, distortions, and ghosting. The wave encoding itself might interact "
                "with off-resonance, potentially requiring advanced correction techniques.")

    def assess_motion_robustness(self) -> str:
        return (f"Standard EPI sequences, which form the basis of Wave-CAIPI ('{self.name}'), "
                "are relatively fast per shot, which can reduce sensitivity to bulk motion within that shot. "
                "However, motion between shots or more complex motion can still be problematic. "
                "Wave encoding itself doesn't inherently add motion robustness beyond the speed of EPI.")

    def suggest_reconstruction_method(self) -> str:
        return (f"Reconstruction for Wave-CAIPI ('{self.name}') requires specialized methods: "
                "1. Unfolding of the CAIPI aliasing, often using sensitivity maps (e.g., SENSE-like reconstruction). "
                "2. Correction for distortions caused by the wave gradients and EPI readouts. "
                "3. Standard EPI corrections (e.g., ghost correction). "
                "Iterative reconstruction methods are common.")
```
