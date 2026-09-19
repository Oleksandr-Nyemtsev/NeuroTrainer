from dataclasses import dataclass


@dataclass
class AudioAction:

    # Binaural layer
    carrier_frequency: float = 180.0
    beat_frequency: float = 8.0
    binaural_volume: float = 0.02

    # Natural soundscape
    wave_volume: float = 0.20
    rain_volume: float = 0.05
    bird_volume: float = 0.00

    # How often natural events occur
    wave_density: float = 0.50
    rain_density: float = 0.10
    bird_density: float = 0.02

    # Overall output
    master_volume: float = 0.70

    # Silence / sparsity
    silence_probability: float = 0.05