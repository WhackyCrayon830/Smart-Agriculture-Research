"""
Enhanced Node class representing an agricultural market (mandi) in India.

Each node represents a physical market location with:
- Price dynamics
- Supply/demand mechanisms
- Neighbor market interactions
- Storage capacity
- Transportation links
"""

import numpy as np


class Node:
    """
    Represents a mandi (agricultural market) node in the network.
    
    Attributes:
        id: Unique identifier
        position: (x, y, z) coordinates
        state: Current market state (price level, normalized)
        neighbors: List of connected market IDs
        
        # Market-specific attributes
        base_price: Baseline equilibrium price
        supply: Current supply level
        demand: Current demand level
        storage: Current storage amount
        storage_capacity: Maximum storage capacity
        
        # Historical tracking
        price_history: Recent price history for momentum calculation
        volatility: Price volatility measure
    """
    
    def __init__(self, node_id, position, base_price=100.0):
        """
        Initialize a mandi node.
        
        Args:
            node_id: Unique identifier
            position: (x, y, z) spatial coordinates
            base_price: Base equilibrium price (default 100.0 rupees/quintal)
        """
        self.id = node_id
        self.position = position
        
        # Core state (normalized price deviation from base)
        self.state = 0.0
        
        # Market economics
        self.base_price = base_price
        self.supply = 1.0  # Normalized supply (1.0 = equilibrium)
        self.demand = 1.0  # Normalized demand (1.0 = equilibrium)
        self.storage = 0.5  # Current storage (0-1 of capacity)
        self.storage_capacity = np.random.uniform(0.8, 1.2)  # Relative capacity
        
        # Network
        self.neighbors = []
        
        # Price dynamics
        self.price_history = [0.0] * 5  # Last 5 states
        self.volatility = 0.1
        
        # Inventory management
        self.last_supply_shock = 0.0
        self.transport_cost_mult = 1.0
    
    @property
    def actual_price(self):
        """
        Get the actual price in rupees.
        
        Returns:
            float: Actual market price (base_price * (1 + state))
        """
        return self.base_price * (1.0 + self.state)
    
    def update_supply_demand(self, weather_effect, time_step):
        """
        Update supply and demand based on weather and time.
        
        Supply affected by:
        - Weather/harvest conditions
        - Seasonal patterns
        - Storage depletion
        
        Demand affected by:
        - Festivals/seasonal consumption
        - Price levels (demand curve)
        - Regional population
        
        Args:
            weather_effect: Weather field value affecting supply
            time_step: Current simulation time
        """
        # Supply: Weather + storage + seasonal harvest
        harvest_season = self._harvest_cycle(time_step)
        
        # Weather reduces supply if negative, increases if positive (up to a point)
        weather_supply = 1.0 + 0.3 * np.tanh(weather_effect)
        
        # Harvest cycle: high supply during harvest, low otherwise
        seasonal_supply = 0.7 + 0.6 * harvest_season
        
        # Storage can buffer supply shortages
        storage_buffer = 0.9 + 0.2 * self.storage
        
        self.supply = weather_supply * seasonal_supply * storage_buffer
        self.supply = np.clip(self.supply, 0.3, 2.0)
        
        # Demand: Festivals + price sensitivity + consumption patterns
        festival_demand = self._festival_cycle(time_step)
        
        # Price elasticity: higher prices reduce demand
        price_elasticity = 1.0 - 0.3 * self.state  # If price up 100%, demand down 30%
        
        # Consumption pattern (slightly random walk)
        base_consumption = 1.0 + 0.1 * np.sin(0.05 * time_step)
        
        self.demand = festival_demand * price_elasticity * base_consumption
        self.demand = np.clip(self.demand, 0.5, 2.0)
        
        # Update storage based on supply-demand balance
        net_flow = self.supply - self.demand
        self.storage += 0.1 * net_flow  # Storage changes slowly
        self.storage = np.clip(self.storage, 0.0, self.storage_capacity)
    
    def _harvest_cycle(self, t):
        """
        Model harvest seasonality (Kharif and Rabi crops).
        
        Kharif: June-October (monsoon crops)
        Rabi: November-March (winter crops)
        
        Args:
            t: Time step (assume 1 step = 1 day)
        
        Returns:
            float: Harvest intensity (0.2 to 1.5)
        """
        day_of_year = (t % 365)
        
        # Kharif harvest: September-October (days 244-304)
        kharif = 0.5 * (1 + np.tanh((day_of_year - 274) / 15)) * (1 - np.tanh((day_of_year - 304) / 15))
        
        # Rabi harvest: March-April (days 60-120)
        rabi = 0.5 * (1 + np.tanh((day_of_year - 60) / 15)) * (1 - np.tanh((day_of_year - 120) / 15))
        
        # Base level + seasonal peaks
        return 0.4 + 0.8 * kharif + 0.6 * rabi
    
    def _festival_cycle(self, t):
        """
        Model festival-driven demand spikes.
        
        Major festivals affecting agricultural markets:
        - Diwali (Oct-Nov): High demand
        - Holi (Mar): Moderate demand
        - Pongal/Makar Sankranti (Jan): High demand
        
        Args:
            t: Time step
        
        Returns:
            float: Festival demand multiplier (0.8 to 1.5)
        """
        day_of_year = (t % 365)
        
        # Diwali region (day 290-310)
        diwali = 0.3 * np.exp(-((day_of_year - 300) ** 2) / 100)
        
        # Pongal (day 14)
        pongal = 0.25 * np.exp(-((day_of_year - 14) ** 2) / 50)
        
        # Holi (day 75)
        holi = 0.15 * np.exp(-((day_of_year - 75) ** 2) / 50)
        
        return 0.9 + diwali + pongal + holi
    
    def calculate_price_pressure(self, neighbor_prices, neighbor_distances):
        """
        Calculate price pressure from neighboring markets.
        
        Markets arbitrage price differences, but with:
        - Distance friction (transport costs)
        - Information lag
        - Transaction costs
        
        Args:
            neighbor_prices: List of neighbor state values
            neighbor_distances: List of distances to neighbors
        
        Returns:
            float: Net price pressure
        """
        if not neighbor_prices:
            return 0.0
        
        total_pressure = 0.0
        total_weight = 0.0
        
        for price, dist in zip(neighbor_prices, neighbor_distances):
            # Weight by inverse distance (closer markets matter more)
            # But with diminishing returns
            weight = 1.0 / (1.0 + dist / 30.0)  # 30 unit half-distance
            
            # Price difference creates arbitrage pressure
            # If neighbor price is higher, upward pressure on our price
            price_diff = price - self.state
            
            # Transport cost reduces arbitrage incentive
            transport_friction = 0.02 * dist * self.transport_cost_mult
            effective_diff = price_diff - np.sign(price_diff) * transport_friction
            
            # Only arbitrage if profitable
            if abs(effective_diff) > transport_friction:
                total_pressure += weight * effective_diff
                total_weight += weight
        
        if total_weight > 0:
            return total_pressure / total_weight
        return 0.0
    
    def update_state(self, weather_effect, neighbor_states, neighbor_distances, 
                     alpha=0.5, beta=0.3, gamma=0.15, delta=0.05):
        """
        Update market state using realistic price dynamics.
        
        Price update equation:
        P(t+1) = α·P(t) + β·weather + γ·neighbors + δ·supply_demand
        
        Args:
            weather_effect: Weather field value
            neighbor_states: List of neighbor price states
            neighbor_distances: List of distances to neighbors
            alpha: Memory/momentum parameter (price stickiness)
            beta: Weather sensitivity
            gamma: Neighbor coupling strength
            delta: Supply-demand sensitivity
        
        Returns:
            float: New state value
        """
        # Store weather for blockchain reporting
        self.last_weather_effect = weather_effect
        
        # 1. Memory: Prices are sticky (don't change instantly)
        memory = alpha * self.state
        
        # 2. Weather: Direct effect on supply/costs
        weather_shock = beta * weather_effect
        
        # 3. Neighbor arbitrage: Price convergence with friction
        neighbor_pressure = self.calculate_price_pressure(
            neighbor_states, 
            neighbor_distances
        )
        neighbor_effect = gamma * neighbor_pressure
        
        # 4. Supply-demand balance
        # If supply > demand, prices fall; if demand > supply, prices rise
        market_imbalance = self.demand - self.supply
        market_effect = delta * market_imbalance
        
        # 5. Storage effect: High storage buffers price increases
        storage_dampening = 1.0 - 0.2 * self.storage
        
        # Combine all effects
        new_state = memory + weather_shock + neighbor_effect + market_effect
        
        # Apply storage dampening to volatility
        new_state *= storage_dampening
        
        # Update volatility (exponential moving average of price changes)
        price_change = abs(new_state - self.state)
        self.volatility = 0.9 * self.volatility + 0.1 * price_change
        
        # Update price history
        self.price_history.pop(0)
        self.price_history.append(self.state)
        
        # Bounded prices (prevent extreme values)
        new_state = np.clip(new_state, -0.8, 1.2)
        
        return new_state
    
    def get_market_info(self):
        """
        Get comprehensive market information for display.
        
        Returns:
            dict: Market statistics
        """
        return {
            'id': self.id,
            'position': self.position,
            'state': self.state,
            'price': self.actual_price,
            'supply': self.supply,
            'demand': self.demand,
            'storage': self.storage,
            'volatility': self.volatility,
            'price_trend': self.state - self.price_history[0],  # 5-step trend
            'market_balance': self.supply - self.demand,
            'num_connections': len(self.neighbors)
        }
    
    def create_blockchain_transaction(self, blockchain, commodity: str = "wheat") -> bool:
        """
        Submit current price to blockchain.
        
        Args:
            blockchain: MandiBlockchain instance
            commodity: Commodity type
        
        Returns:
            bool: True if transaction accepted
        """
        # Estimate trading volume based on market activity
        trading_volume = self.demand * (1.0 - abs(self.storage - 0.5))  # Higher when balanced
        
        # Determine weather condition
        if hasattr(self, 'last_weather_effect'):
            if self.last_weather_effect > 0.5:
                weather = "heavy_rain"
            elif self.last_weather_effect > 0:
                weather = "moderate_rain"
            elif self.last_weather_effect > -0.5:
                weather = "clear"
            else:
                weather = "drought"
        else:
            weather = "unknown"
        
        tx = blockchain.submit_price_transaction(
            mandi_id=self.id,
            price=self.actual_price,
            volume=trading_volume,
            supply=self.supply,
            demand=self.demand,
            weather_condition=weather,
            commodity=commodity
        )
        
        return tx is not None
    
    def get_blockchain_price_history(self, blockchain, lookback: int = 100):
        """
        Get this mandi's price history from blockchain.
        
        Args:
            blockchain: MandiBlockchain instance
            lookback: Number of blocks to look back
        
        Returns:
            List of (timestamp, price) tuples
        """
        return blockchain.get_mandi_price_history(self.id, lookback)
    
    def __repr__(self):
        return (f"Mandi({self.id}, price={self.actual_price:.1f}, "
                f"supply={self.supply:.2f}, demand={self.demand:.2f})")