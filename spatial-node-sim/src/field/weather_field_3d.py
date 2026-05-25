"""
Enhanced 3D Weather Field for Agricultural Market (Mandi) Simulation

This module simulates realistic weather patterns that affect agricultural markets in India:
- Monsoon patterns with seasonal cycles
- Temperature variations
- Rainfall intensity
- Drought/flood events
- Regional weather systems

The weather field influences crop yields, transportation costs, and market prices.
"""

import numpy as np


class WeatherField3D:
    """
    Realistic weather simulation for Indian agricultural markets.
    
    Combines multiple weather phenomena:
    - Seasonal monsoon cycles (June-September peak)
    - Temperature patterns
    - Rainfall cells that move spatially
    - Extreme weather events (droughts, floods)
    - Day/night cycles
    
    Attributes:
        seed: Random seed for reproducibility
        num_centers: Number of weather system centers
        monsoon_strength: Base strength of monsoon effect (0-1)
        extreme_event_prob: Probability of extreme weather per timestep
    """
    
    def __init__(self, seed=None, num_centers=6, monsoon_strength=0.7):
        """
        Initialize the weather field with realistic Indian weather patterns.
        
        Args:
            seed: Random seed for reproducibility
            num_centers: Number of weather system centers (default 6 for India's climate zones)
            monsoon_strength: Strength of monsoon effect (0-1, default 0.7)
        """
        self.rng = np.random.default_rng(seed)
        self.num_centers = num_centers
        self.monsoon_strength = monsoon_strength
        
        # Weather system centers (representing low/high pressure systems)
        self.centers = self.rng.uniform(0, 100, (num_centers, 3))
        
        # Spatial spread of weather systems (km equivalent)
        self.sigmas = self.rng.uniform(15, 40, num_centers)
        
        # Base intensities of weather systems
        self.amplitudes = self.rng.uniform(0.5, 2.5, num_centers)
        
        # Movement speeds (how fast weather systems evolve)
        self.omegas = self.rng.uniform(0.02, 0.08, num_centers)
        
        # Phase offsets for temporal variation
        self.phases = self.rng.uniform(0, 2*np.pi, num_centers)
        
        # Movement directions for weather cells
        self.velocities = self.rng.uniform(-0.5, 0.5, (num_centers, 3))
        
        # Extreme event tracking
        self.last_extreme_event = -100
        self.current_extreme = None
        self.extreme_duration = 0
    
    def _monsoon_cycle(self, t):
        """
        Calculate monsoon intensity based on annual cycle.
        
        Indian monsoon pattern:
        - Weak: Jan-May (summer heat builds)
        - Strong: June-September (monsoon season)
        - Moderate: October-December (retreating monsoon)
        
        Args:
            t: Time step (assume 1 step = 1 day for realism)
        
        Returns:
            float: Monsoon intensity (0-1)
        """
        # Annual cycle (365 days)
        day_of_year = (t % 365)
        
        # Peak during June-September (days 152-273)
        # Using smooth transitions
        if 152 <= day_of_year <= 273:
            # Monsoon season - high intensity
            return 0.8 + 0.2 * np.sin(2 * np.pi * (day_of_year - 152) / 121)
        elif 274 <= day_of_year <= 365 or day_of_year <= 60:
            # Post-monsoon and winter - moderate
            return 0.3 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
        else:
            # Pre-monsoon summer - low rainfall, high heat
            return 0.1 + 0.15 * np.sin(2 * np.pi * day_of_year / 365)
    
    def _temperature_effect(self, t):
        """
        Calculate temperature variation effect.
        
        Temperature affects:
        - Crop growth rates
        - Storage costs
        - Transportation feasibility
        
        Args:
            t: Time step
        
        Returns:
            float: Temperature effect (-1 to 1)
        """
        # Annual temperature cycle
        day_of_year = (t % 365)
        
        # Daily cycle (faster oscillation)
        daily = 0.3 * np.sin(2 * np.pi * t / 1)
        
        # Seasonal cycle (peak in May, low in January)
        seasonal = np.sin(2 * np.pi * (day_of_year - 120) / 365)
        
        return 0.6 * seasonal + 0.4 * daily
    
    def _extreme_event(self, t):
        """
        Simulate extreme weather events (droughts, floods, heatwaves).
        
        Args:
            t: Current timestep
        
        Returns:
            float: Extreme event contribution
        """
        # Check if we should start a new extreme event
        if self.current_extreme is None and t - self.last_extreme_event > 50:
            if self.rng.random() < 0.01:  # 1% chance per timestep
                self.current_extreme = self.rng.choice(['drought', 'flood', 'heatwave'])
                self.extreme_duration = self.rng.integers(10, 30)
                self.last_extreme_event = t
        
        # Apply current extreme event
        if self.current_extreme is not None:
            self.extreme_duration -= 1
            if self.extreme_duration <= 0:
                self.current_extreme = None
                return 0.0
            
            if self.current_extreme == 'drought':
                return -0.8  # Severe negative effect
            elif self.current_extreme == 'flood':
                return 0.9  # Severe positive (too much water)
            elif self.current_extreme == 'heatwave':
                return -0.5  # Moderate negative
        
        return 0.0
    
    def value(self, pos, t):
        """
        Calculate weather field value at a position and time.
        
        Combines:
        - Multiple weather system centers
        - Seasonal monsoon patterns
        - Temperature effects
        - Extreme weather events
        - Spatial and temporal coherence
        
        Args:
            pos: 3D position array [x, y, z]
            t: Time step
        
        Returns:
            float: Weather field value (typically -2 to 2)
                  Positive: High rainfall/good conditions
                  Negative: Drought/poor conditions
        """
        total = 0.0
        
        # Multi-center weather systems
        for i, (A, mu, sig, w, phi, vel) in enumerate(zip(
            self.amplitudes, 
            self.centers, 
            self.sigmas, 
            self.omegas, 
            self.phases,
            self.velocities
        )):
            # Move weather center over time (weather systems drift)
            current_center = mu + vel * t * 0.1
            
            # Spatial component: Gaussian falloff from center
            spatial_dist = np.sum((pos - current_center) ** 2) / (2 * sig ** 2)
            spatial = np.exp(-spatial_dist)
            
            # Temporal component: Oscillation with phase
            temporal = np.cos(w * t + phi)
            
            # Combine
            total += A * spatial * temporal
        
        # Add monsoon cycle modulation
        monsoon = self._monsoon_cycle(t)
        total *= (0.5 + 0.5 * monsoon * self.monsoon_strength)
        
        # Add temperature effect
        temp = self._temperature_effect(t)
        total += 0.3 * temp
        
        # Add extreme events
        extreme = self._extreme_event(t)
        total += extreme
        
        # Add small-scale turbulence/noise
        noise = self.rng.normal(0, 0.05)
        total += noise
        
        return total
    
    def get_status(self, t):
        """
        Get human-readable weather status.
        
        Args:
            t: Current timestep
        
        Returns:
            dict: Weather status information
        """
        monsoon = self._monsoon_cycle(t)
        temp = self._temperature_effect(t)
        
        # Determine season
        day_of_year = (t % 365)
        if 152 <= day_of_year <= 273:
            season = "Monsoon"
        elif 274 <= day_of_year <= 365 or day_of_year <= 60:
            season = "Post-Monsoon/Winter"
        else:
            season = "Summer"
        
        return {
            'season': season,
            'day_of_year': day_of_year,
            'monsoon_intensity': monsoon,
            'temperature_effect': temp,
            'extreme_event': self.current_extreme,
            'extreme_days_remaining': self.extreme_duration if self.current_extreme else 0
        }


class MarketWeatherField(WeatherField3D):
    """
    Extension of WeatherField3D specifically for mandi price modeling.
    
    Adds market-specific interpretations:
    - Converts weather to supply shocks
    - Models transportation disruptions
    - Calculates storage condition effects
    """
    
    def supply_shock(self, pos, t):
        """
        Convert weather to supply availability shock.
        
        Args:
            pos: Market position
            t: Time step
        
        Returns:
            float: Supply multiplier (0.5 = 50% supply, 2.0 = 200% supply)
        """
        weather = self.value(pos, t)
        
        # Good weather (moderate positive) = good supply
        # Too much rain or drought = poor supply
        if -0.3 < weather < 0.5:
            # Optimal weather
            supply_mult = 1.0 + 0.5 * weather
        elif weather < -0.3:
            # Drought or poor conditions
            supply_mult = 0.5 + 0.3 * (weather + 1)
        else:
            # Excessive rain/flooding
            supply_mult = 1.0 - 0.3 * (weather - 0.5)
        
        return np.clip(supply_mult, 0.3, 1.8)
    
    def transport_cost(self, pos, t):
        """
        Calculate transportation cost multiplier based on weather.
        
        Args:
            pos: Market position
            t: Time step
        
        Returns:
            float: Cost multiplier (1.0 = normal, >1 = higher cost)
        """
        weather = self.value(pos, t)
        
        # Heavy rain increases transport costs
        # Extreme heat also increases costs (spoilage)
        extreme = abs(weather)
        
        if extreme > 1.0:
            return 1.0 + 0.5 * extreme  # Up to 50% cost increase
        
        return 1.0 + 0.1 * extreme