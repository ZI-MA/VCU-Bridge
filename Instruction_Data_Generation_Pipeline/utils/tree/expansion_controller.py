import threading
from typing import Dict, Any
from ..nodes import MCTSNode


class TreeExpansionController:
    """Simplified tree expansion controller"""
    
    def __init__(self, config: Dict[str, Any]):
        tree_config = config.get('tree', {})

        # Hard constraint parameters
        max_children_config = tree_config.get('max_children', 5)
        if isinstance(max_children_config, dict):
            self.max_children_per_level = max_children_config
        else:
            self.max_children_per_level = {}
        self.max_children_default = max_children_config if not isinstance(max_children_config, dict) else 5

        self.max_depth = config.get('mcts', {}).get('max_depth', 3)
        self.max_total_nodes = tree_config.get('max_total_nodes', 80)
        self.level_configs = tree_config.get('levels', {})
        
        # Quality strategy parameters
        quality_config = tree_config.get('quality', {})
        self.quality_thresholds = quality_config.get('thresholds', {"high": 0.8, "medium": 0.5})
        self.quality_multipliers = quality_config.get('quality_multipliers', {"high": 1.2, "medium": 1.0, "low": 0.8})
        
        # Dynamic adjustment parameters (optional)
        dynamic_config = tree_config.get('dynamic', {})
        self.enable_dynamic = dynamic_config.get('enabled', False)
        self.dynamic_weights = dynamic_config.get('weights', {"quality": 0.5, "balance": 0.3, "success": 0.2})
        self.adjustment_range = dynamic_config.get('adjustment_range', [0.7, 1.3])
        
        self._last_limiting_factor = None
        # Use reentrant lock to avoid self-deadlock during selection
        self._expansion_lock = threading.RLock()
    
    def get_expansion_count(self, node: MCTSNode, tree, allocated_per_level: dict = None, allocated_total: int = 0) -> int:
        """Get suggested expansion count"""
        # Hard constraint check (considering allocated capacity)
        hard_limit = self._calculate_hard_constraints(node, tree, allocated_per_level, allocated_total)
        if hard_limit <= 0:
            return 0
        
        # Quality adjustment
        quality_adjusted = self._apply_quality_strategy(node, hard_limit)
        
        # Dynamic adjustment (optional)
        if self.enable_dynamic:
            return self._apply_dynamic_strategy(node, tree, quality_adjusted)
        
        return quality_adjusted
    
    def can_expand(self, node: MCTSNode, tree) -> bool:
        """Quick check if expansion is possible"""
        return self._calculate_hard_constraints(node, tree, None, 0) > 0
    
    def get_expansion_info(self, node: MCTSNode, tree) -> Dict[str, Any]:
        """Get detailed expansion info"""
        hard_limit = self._calculate_hard_constraints(node, tree, None, 0)
        quality_adjusted = self._apply_quality_strategy(node, hard_limit) if hard_limit > 0 else 0
        final_count = self.get_expansion_count(node, tree)
        
        return {
            "hard_limit": hard_limit,
            "quality_adjusted": quality_adjusted,
            "final_count": final_count,
            "limiting_factor": self._last_limiting_factor,
            "node_quality": node.average_reward,
            "dynamic_enabled": self.enable_dynamic
        }
    
    def _get_max_children_for_level(self, parent_level: int) -> int:
        """Get max children for specific parent level"""
        if parent_level == 0:
            level_config = self.level_configs.get('1', {})
            return level_config.get('target_nodes', self.max_children_default)

        child_level = parent_level + 1
        if child_level > self.max_depth:
            return 0

        level_key = str(parent_level)
        return self.max_children_per_level.get(level_key, self.max_children_default)

    def _calculate_hard_constraints(self, node: MCTSNode, tree, allocated_per_level: dict = None, allocated_total: int = 0) -> int:
        """Calculate hard constraints"""
        with self._expansion_lock:
            if allocated_per_level is None:
                allocated_per_level = {}
                
            target_level = node.level + 1
            constraints = []

            # Node children limit (dynamic by level)
            max_children = self._get_max_children_for_level(node.level)
            node_capacity = max_children - len(node.children)
            constraints.append(("node", node_capacity))
            
            # Target level capacity limit (minus allocated)
            level_nodes = len(tree.get_nodes_by_level(target_level))
            level_max = self._get_level_max(target_level)
            level_allocated = allocated_per_level.get(target_level, 0)
            level_capacity = level_max - level_nodes - level_allocated
            constraints.append(("level", level_capacity))
            
            # Tree total nodes limit (minus allocated)
            total_nodes = len([n for n in tree.nodes.values() if n.level > 0])
            tree_capacity = self.max_total_nodes - total_nodes - allocated_total
            constraints.append(("tree", tree_capacity))
            
            # Find minimum constraint
            min_constraint = min(constraints, key=lambda x: x[1])
            self._last_limiting_factor = min_constraint[0]
            
            return max(0, min_constraint[1])
    
    def _get_level_max(self, level: int) -> int:
        """Get max nodes for level"""
        level_config = self.level_configs.get(str(level), {})
        return level_config.get('max_nodes', 100)
    
    def _apply_quality_strategy(self, node: MCTSNode, max_count: int) -> int:
        """Adjust expansion count based on node quality"""
        if max_count <= 0:
            return 0
        
        # Get target nodes for level as base
        target_level = node.level + 1
        level_config = self.level_configs.get(str(target_level), {})
        base_target = level_config.get('target_nodes', 2)
        
        if node.level == 0:
            return min(base_target, max_count)
        
        # Get multiplier based on quality
        quality = node.average_reward
        if quality >= self.quality_thresholds["high"]:
            multiplier = self.quality_multipliers["high"]
        elif quality >= self.quality_thresholds["medium"]:
            multiplier = self.quality_multipliers["medium"]  
        else:
            multiplier = self.quality_multipliers["low"]
        
        # Calculate adjusted suggestion
        suggestion = max(1, int(base_target * multiplier))
        return min(max_count, suggestion)
    
    def _apply_dynamic_strategy(self, node: MCTSNode, tree, base_count: int) -> int:
        """Dynamic adjustment strategy (optional)"""
        if base_count <= 0:
            return 0
        
        # Simplified performance analysis
        nodes = [n for n in tree.nodes.values() if n.level > 0]
        if not nodes:
            return base_count
            
        # Calculate overall performance metrics
        avg_quality = sum(n.average_reward for n in nodes) / len(nodes)
        success_rate = min(1.0, len(nodes) / max(1, tree.iteration_count))
        
        # Calculate balance score
        level_counts = {}
        for n in nodes:
            level_counts[n.level] = level_counts.get(n.level, 0) + 1
        
        balance_score = 0.5
        if len(level_counts) > 1:
            vals = list(level_counts.values())
            mean_val = sum(vals) / len(vals)
            variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
            balance_score = max(0.0, 1.0 - (variance / mean_val if mean_val > 0 else 1.0))
        
        # Calculate weighted overall score
        overall_score = (
            avg_quality * self.dynamic_weights["quality"] +
            success_rate * self.dynamic_weights["success"] +
            balance_score * self.dynamic_weights["balance"]
        )
        
        # Calculate adjustment multiplier
        if overall_score > 0.7:
            multiplier = min(self.adjustment_range[1], 1.2)
        elif overall_score < 0.4:
            multiplier = max(self.adjustment_range[0], 0.8)
        else:
            multiplier = 1.0
        
        adjusted_count = max(1, int(base_count * multiplier))
        return max(base_count - 2, min(adjusted_count, base_count + 2))
