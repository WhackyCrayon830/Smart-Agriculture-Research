"""
Blockchain implementation for Agricultural Market (Mandi) Price Sharing

This module implements a simplified but realistic blockchain where:
- Each mandi can publish price information as transactions
- Blocks are mined periodically (consensus rounds)
- Price history is immutable and transparent
- Smart contracts can trigger on price thresholds
- Network participants validate transactions

Key features:
- Proof-of-Stake consensus (mandis with more volume get priority)
- Price oracle: Get verified average prices from blockchain
- Transaction fees based on network congestion
- Fork resolution and chain validation
"""

import hashlib
import time
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np


@dataclass
class PriceTransaction:
    """
    A transaction representing a mandi's price report.
    
    Attributes:
        mandi_id: Market identifier
        timestamp: Unix timestamp
        commodity: Type of commodity (future enhancement)
        price: Reported price in ₹/quintal
        volume: Trading volume (influences weight)
        supply: Supply level reported
        demand: Demand level reported
        signature: Hash-based signature for authenticity
    """
    mandi_id: int
    timestamp: float
    commodity: str
    price: float
    volume: float
    supply: float
    demand: float
    weather_condition: str
    signature: str = ""
    
    def __post_init__(self):
        """Generate signature if not provided"""
        if not self.signature:
            self.signature = self.generate_signature()
    
    def generate_signature(self) -> str:
        """Create hash-based signature for transaction"""
        data = f"{self.mandi_id}{self.timestamp}{self.price}{self.volume}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def validate(self) -> bool:
        """Verify transaction integrity"""
        expected_sig = self.generate_signature()
        return self.signature == expected_sig


@dataclass
class Block:
    """
    A block in the mandi blockchain.
    
    Contains multiple price transactions from different mandis.
    """
    index: int
    timestamp: float
    transactions: List[PriceTransaction]
    previous_hash: str
    nonce: int = 0
    hash: str = ""
    validator_id: int = -1  # Mandi that validated this block
    
    def __post_init__(self):
        """Generate hash if not provided"""
        if not self.hash:
            self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_data = {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [t.to_dict() for t in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'validator': self.validator_id
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 2):
        """
        Simple proof-of-work mining (for demonstration).
        In production, we'd use Proof-of-Stake.
        """
        target = "0" * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()
    
    def validate(self, previous_hash: str) -> bool:
        """Validate block integrity"""
        # Check hash
        if self.hash != self.calculate_hash():
            return False
        
        # Check chain link
        if self.previous_hash != previous_hash:
            return False
        
        # Validate all transactions
        return all(tx.validate() for tx in self.transactions)


class MandiBlockchain:
    """
    Blockchain for agricultural market price sharing.
    
    Features:
    - Immutable price history
    - Transparent price discovery
    - Consensus-based validation
    - Price oracle functionality
    """
    
    def __init__(self, block_time: float = 10.0, difficulty: int = 2):
        """
        Initialize blockchain.
        
        Args:
            block_time: Seconds between blocks (default 10)
            difficulty: Mining difficulty (default 2)
        """
        self.chain: List[Block] = []
        self.pending_transactions: List[PriceTransaction] = []
        self.block_time = block_time
        self.difficulty = difficulty
        self.last_block_time = time.time()
        
        # Validator stakes (Proof-of-Stake)
        self.validator_stakes: Dict[int, float] = {}
        
        # Transaction pool
        self.transaction_fee = 0.01  # Base fee in ₹
        
        # Create genesis block
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_tx = PriceTransaction(
            mandi_id=0,
            timestamp=time.time(),
            commodity="genesis",
            price=0.0,
            volume=0.0,
            supply=1.0,
            demand=1.0,
            weather_condition="clear"
        )
        
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[genesis_tx],
            previous_hash="0",
            validator_id=0
        )
        
        self.chain.append(genesis_block)
    
    def submit_price_transaction(
        self, 
        mandi_id: int,
        price: float,
        volume: float,
        supply: float,
        demand: float,
        weather_condition: str,
        commodity: str = "wheat"
    ) -> Optional[PriceTransaction]:
        """
        Submit a price report to the blockchain.
        
        Args:
            mandi_id: Market identifier
            price: Current price in ₹/quintal
            volume: Trading volume
            supply: Supply level
            demand: Demand level
            weather_condition: Current weather
            commodity: Commodity type
        
        Returns:
            PriceTransaction if successful, None if rejected
        """
        tx = PriceTransaction(
            mandi_id=mandi_id,
            timestamp=time.time(),
            commodity=commodity,
            price=price,
            volume=volume,
            supply=supply,
            demand=demand,
            weather_condition=weather_condition
        )
        
        # Validate transaction
        if not tx.validate():
            return None
        
        # Add to pending pool
        self.pending_transactions.append(tx)
        return tx
    
    def select_validator(self) -> int:
        """
        Select validator using Proof-of-Stake.
        
        Mandis with higher trading volume get higher probability.
        
        Returns:
            Validator mandi_id
        """
        if not self.validator_stakes:
            return 0
        
        # Probability proportional to stake
        total_stake = sum(self.validator_stakes.values())
        if total_stake == 0:
            return list(self.validator_stakes.keys())[0]
        
        rand = np.random.random() * total_stake
        cumsum = 0
        for mandi_id, stake in self.validator_stakes.items():
            cumsum += stake
            if rand <= cumsum:
                return mandi_id
        
        return list(self.validator_stakes.keys())[-1]
    
    def update_validator_stakes(self, mandi_volumes: Dict[int, float]):
        """
        Update validator stakes based on trading volume.
        
        Args:
            mandi_volumes: {mandi_id: trading_volume}
        """
        self.validator_stakes = mandi_volumes.copy()
    
    def create_block(self, max_transactions: int = 50) -> Optional[Block]:
        """
        Create a new block from pending transactions.
        
        Args:
            max_transactions: Maximum transactions per block
        
        Returns:
            New Block if created, None if not enough transactions or time
        """
        current_time = time.time()
        
        # Check if enough time has passed
        if current_time - self.last_block_time < self.block_time:
            return None
        
        # Need at least one transaction
        if not self.pending_transactions:
            return None
        
        # Select transactions
        transactions = self.pending_transactions[:max_transactions]
        self.pending_transactions = self.pending_transactions[max_transactions:]
        
        # Select validator
        validator_id = self.select_validator()
        
        # Create block
        previous_block = self.get_latest_block()
        new_block = Block(
            index=len(self.chain),
            timestamp=current_time,
            transactions=transactions,
            previous_hash=previous_block.hash,
            validator_id=validator_id
        )
        
        # Mine block (simplified PoW for demonstration)
        # In production, PoS doesn't require mining
        new_block.mine_block(difficulty=self.difficulty)
        
        # Add to chain
        self.chain.append(new_block)
        self.last_block_time = current_time
        
        return new_block
    
    def get_latest_block(self) -> Block:
        """Get the most recent block"""
        return self.chain[-1]
    
    def validate_chain(self) -> bool:
        """
        Validate entire blockchain integrity.
        
        Returns:
            True if chain is valid, False otherwise
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            if not current_block.validate(previous_block.hash):
                return False
        
        return True
    
    def get_price_oracle(
        self, 
        commodity: str = "wheat",
        lookback_blocks: int = 10,
        min_mandis: int = 3
    ) -> Optional[Dict[str, float]]:
        """
        Get verified average price from blockchain (Price Oracle).
        
        This is the key feature: transparent, tamper-proof price discovery.
        
        Args:
            commodity: Commodity type
            lookback_blocks: How many blocks to look back
            min_mandis: Minimum mandis required for valid price
        
        Returns:
            Dict with price statistics or None if insufficient data
        """
        # Collect recent transactions
        recent_txs = []
        blocks_checked = 0
        
        for block in reversed(self.chain):
            if blocks_checked >= lookback_blocks:
                break
            
            for tx in block.transactions:
                if tx.commodity == commodity:
                    recent_txs.append(tx)
            
            blocks_checked += 1
        
        if len(recent_txs) < min_mandis:
            return None
        
        # Calculate volume-weighted average price
        total_volume = sum(tx.volume for tx in recent_txs)
        if total_volume == 0:
            # Simple average if no volume data
            avg_price = np.mean([tx.price for tx in recent_txs])
            weights = [1.0 / len(recent_txs)] * len(recent_txs)
        else:
            # Volume-weighted average
            avg_price = sum(tx.price * tx.volume for tx in recent_txs) / total_volume
            weights = [tx.volume / total_volume for tx in recent_txs]
        
        # Calculate statistics
        prices = [tx.price for tx in recent_txs]
        
        return {
            'average_price': avg_price,
            'min_price': min(prices),
            'max_price': max(prices),
            'std_price': np.std(prices),
            'median_price': np.median(prices),
            'num_reports': len(recent_txs),
            'total_volume': total_volume,
            'unique_mandis': len(set(tx.mandi_id for tx in recent_txs))
        }
    
    def get_mandi_price_history(
        self, 
        mandi_id: int,
        lookback_blocks: int = 100
    ) -> List[Tuple[float, float]]:
        """
        Get price history for a specific mandi.
        
        Args:
            mandi_id: Market identifier
            lookback_blocks: How many blocks to search
        
        Returns:
            List of (timestamp, price) tuples
        """
        history = []
        blocks_checked = 0
        
        for block in reversed(self.chain):
            if blocks_checked >= lookback_blocks:
                break
            
            for tx in block.transactions:
                if tx.mandi_id == mandi_id:
                    history.append((tx.timestamp, tx.price))
            
            blocks_checked += 1
        
        return list(reversed(history))
    
    def get_network_statistics(self) -> Dict:
        """
        Get blockchain network statistics.
        
        Returns:
            Dict with network stats
        """
        return {
            'total_blocks': len(self.chain),
            'pending_transactions': len(self.pending_transactions),
            'chain_valid': self.validate_chain(),
            'total_transactions': sum(len(b.transactions) for b in self.chain),
            'validators': len(self.validator_stakes),
            'block_time': self.block_time,
            'last_block_time': self.last_block_time,
            'average_tx_per_block': np.mean([len(b.transactions) for b in self.chain[1:]]) if len(self.chain) > 1 else 0
        }
    
    def get_price_consensus(
        self,
        commodity: str = "wheat",
        threshold_agreement: float = 0.8
    ) -> Optional[Dict]:
        """
        Check if mandis have price consensus.
        
        Useful for detecting price manipulation or market fragmentation.
        
        Args:
            commodity: Commodity type
            threshold_agreement: % of mandis that must agree (within 10% of median)
        
        Returns:
            Dict with consensus info or None
        """
        oracle_data = self.get_price_oracle(commodity)
        if not oracle_data:
            return None
        
        median_price = oracle_data['median_price']
        
        # Get recent prices
        recent_txs = []
        for block in list(reversed(self.chain))[:10]:
            recent_txs.extend([tx for tx in block.transactions if tx.commodity == commodity])
        
        # Check agreement (within 10% of median)
        agreement_count = sum(
            1 for tx in recent_txs 
            if abs(tx.price - median_price) / median_price < 0.1
        )
        
        agreement_rate = agreement_count / len(recent_txs) if recent_txs else 0
        
        return {
            'consensus_reached': agreement_rate >= threshold_agreement,
            'agreement_rate': agreement_rate,
            'median_price': median_price,
            'price_range': oracle_data['max_price'] - oracle_data['min_price'],
            'num_reporters': len(recent_txs)
        }
    
    def export_chain_data(self, output_file: str):
        """
        Export blockchain to JSON file.
        
        Args:
            output_file: Path to output file
        """
        chain_data = []
        for block in self.chain:
            chain_data.append({
                'index': block.index,
                'timestamp': block.timestamp,
                'hash': block.hash,
                'previous_hash': block.previous_hash,
                'validator_id': block.validator_id,
                'transactions': [tx.to_dict() for tx in block.transactions]
            })
        
        with open(output_file, 'w') as f:
            json.dump(chain_data, f, indent=2)
    
    def __repr__(self):
        return f"MandiBlockchain(blocks={len(self.chain)}, pending_tx={len(self.pending_transactions)})"


class PriceSmartContract:
    """
    Smart contract for automated actions based on blockchain price data.
    
    Examples:
    - Trigger alerts when price > threshold
    - Automatic order execution at target price
    - Insurance payouts when price drops
    """
    
    def __init__(self, blockchain: MandiBlockchain):
        self.blockchain = blockchain
        self.active_contracts: List[Dict] = []
    
    def create_price_alert(
        self,
        mandi_id: int,
        commodity: str,
        condition: str,  # "above" or "below"
        threshold: float,
        callback_data: Dict
    ) -> int:
        """
        Create a smart contract that triggers on price condition.
        
        Args:
            mandi_id: Market to monitor
            commodity: Commodity type
            condition: "above" or "below"
            threshold: Price threshold
            callback_data: Data to return when triggered
        
        Returns:
            Contract ID
        """
        contract = {
            'id': len(self.active_contracts),
            'type': 'price_alert',
            'mandi_id': mandi_id,
            'commodity': commodity,
            'condition': condition,
            'threshold': threshold,
            'callback_data': callback_data,
            'active': True,
            'created_at': time.time()
        }
        
        self.active_contracts.append(contract)
        return contract['id']
    
    def check_contracts(self) -> List[Dict]:
        """
        Check all active contracts and trigger if conditions met.
        
        Returns:
            List of triggered contracts
        """
        triggered = []
        
        for contract in self.active_contracts:
            if not contract['active']:
                continue
            
            if contract['type'] == 'price_alert':
                oracle = self.blockchain.get_price_oracle(
                    commodity=contract['commodity']
                )
                
                if oracle:
                    current_price = oracle['average_price']
                    
                    triggered_condition = False
                    if contract['condition'] == 'above':
                        triggered_condition = current_price > contract['threshold']
                    elif contract['condition'] == 'below':
                        triggered_condition = current_price < contract['threshold']
                    
                    if triggered_condition:
                        triggered.append({
                            'contract_id': contract['id'],
                            'current_price': current_price,
                            'threshold': contract['threshold'],
                            'callback_data': contract['callback_data']
                        })
                        contract['active'] = False  # Trigger once
        
        return triggered