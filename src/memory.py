# memory.py

import re
import math
import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Set, Union, Tuple, Callable
from dataclasses import dataclass, field
from copy import deepcopy
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
import heapq


# ============================================================
# Part 1: Original Memory Processor (Think/Tool Call Processing)
# ============================================================

class ThinkRemovalMode(Enum):
    """思考块移除模式"""
    REMOVE_ALL = "remove_all"
    KEEP_LAST = "keep_last"
    KEEP_ALL = "keep_all"
    REMOVE_PERCENT = "remove_percent"


class ToolCallRemovalMode(Enum):
    """工具调用块移除模式"""
    REMOVE_ALL = "remove_all"
    KEEP_LAST = "keep_last"
    KEEP_ALL = "keep_all"
    REMOVE_PERCENT = "remove_percent"


@dataclass
class MemoryProcessor:
    """
    消息预处理器，负责在消息传入 Agent 之前进行处理。
    """
    think_mode: ThinkRemovalMode = ThinkRemovalMode.REMOVE_ALL
    tool_call_mode: ToolCallRemovalMode = ToolCallRemovalMode.KEEP_ALL
    think_remove_percent: float = 1.0
    tool_call_remove_percent: float = 1.0
    
    think_pattern: re.Pattern = field(
        default_factory=lambda: re.compile(r'<think>.*?</think>', re.DOTALL)
    )
    tool_call_pattern: re.Pattern = field(
        default_factory=lambda: re.compile(r'<tool_call>.*?</tool_call>', re.DOTALL)
    )
    roles_to_process: Set[str] = field(default_factory=lambda: {'assistant'})
    
    def __post_init__(self):
        if not 0.0 <= self.think_remove_percent <= 1.0:
            raise ValueError(f"think_remove_percent must be between 0.0 and 1.0")
        if not 0.0 <= self.tool_call_remove_percent <= 1.0:
            raise ValueError(f"tool_call_remove_percent must be between 0.0 and 1.0")
    
    def _find_all_blocks_in_message(self, content: str, pattern: re.Pattern) -> List[re.Match]:
        if not content:
            return []
        return list(pattern.finditer(content))
    
    def _remove_all_blocks(self, content: str, pattern: re.Pattern) -> str:
        if not content:
            return content
        return pattern.sub('', content)
    
    def _clean_whitespace(self, content: str) -> str:
        cleaned = re.sub(r'\n{3,}', '\n\n', content)
        return cleaned.strip()
    
    def _find_all_blocks_across_messages(
        self, messages: List[Dict[str, Any]], pattern: re.Pattern
    ) -> List[tuple]:
        all_blocks = []
        for msg_idx, msg in enumerate(messages):
            if msg.get('role') not in self.roles_to_process:
                continue
            content = msg.get('content', '')
            if not isinstance(content, str):
                continue
            for match in pattern.finditer(content):
                all_blocks.append((msg_idx, match))
        return all_blocks
    
    def _process_messages_remove_all(
        self, messages: List[Dict[str, Any]], pattern: re.Pattern
    ) -> List[Dict[str, Any]]:
        result = []
        for msg in messages:
            processed = deepcopy(msg)
            if msg.get('role') in self.roles_to_process:
                content = msg.get('content', '')
                if isinstance(content, str):
                    processed['content'] = self._remove_all_blocks(content, pattern)
                    processed['content'] = self._clean_whitespace(processed['content'])
            result.append(processed)
        return result
    
    def _process_messages_keep_last(
        self, messages: List[Dict[str, Any]], pattern: re.Pattern
    ) -> List[Dict[str, Any]]:
        all_blocks = self._find_all_blocks_across_messages(messages, pattern)
        if len(all_blocks) <= 1:
            return deepcopy(messages)
        blocks_to_remove = all_blocks[:-1]
        return self._remove_specified_blocks(messages, blocks_to_remove)
    
    def _process_messages_remove_percent(
        self, messages: List[Dict[str, Any]], pattern: re.Pattern, percent: float
    ) -> List[Dict[str, Any]]:
        if percent <= 0:
            return deepcopy(messages)
        if percent >= 1.0:
            return self._process_messages_remove_all(messages, pattern)
        
        all_blocks = self._find_all_blocks_across_messages(messages, pattern)
        if len(all_blocks) == 0:
            return deepcopy(messages)
        
        num_to_remove = math.ceil(len(all_blocks) * percent)
        if num_to_remove == 0:
            return deepcopy(messages)
        if num_to_remove >= len(all_blocks):
            return self._process_messages_remove_all(messages, pattern)
        
        blocks_to_remove = all_blocks[:num_to_remove]
        return self._remove_specified_blocks(messages, blocks_to_remove)
    
    def _remove_specified_blocks(
        self, messages: List[Dict[str, Any]], blocks_to_remove: List[tuple]
    ) -> List[Dict[str, Any]]:
        if not blocks_to_remove:
            return deepcopy(messages)
        
        remove_by_msg: Dict[int, List[re.Match]] = {}
        for msg_idx, match in blocks_to_remove:
            if msg_idx not in remove_by_msg:
                remove_by_msg[msg_idx] = []
            remove_by_msg[msg_idx].append(match)
        
        result = []
        for msg_idx, msg in enumerate(messages):
            processed = deepcopy(msg)
            if msg.get('role') in self.roles_to_process:
                content = msg.get('content', '')
                if isinstance(content, str) and msg_idx in remove_by_msg:
                    matches_to_remove = sorted(
                        remove_by_msg[msg_idx], 
                        key=lambda m: m.start(), 
                        reverse=True
                    )
                    for match in matches_to_remove:
                        start, end = match.span()
                        content = content[:start] + content[end:]
                    processed['content'] = self._clean_whitespace(content)
            result.append(processed)
        return result
    
    def _process_for_block_type(
        self, messages: List[Dict[str, Any]], pattern: re.Pattern, 
        mode: Enum, percent: float = 1.0
    ) -> List[Dict[str, Any]]:
        if mode.value == "keep_all":
            return deepcopy(messages)
        elif mode.value == "remove_all":
            return self._process_messages_remove_all(messages, pattern)
        elif mode.value == "keep_last":
            return self._process_messages_keep_last(messages, pattern)
        elif mode.value == "remove_percent":
            return self._process_messages_remove_percent(messages, pattern, percent)
        return deepcopy(messages)
    
    def process_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = self._process_for_block_type(
            messages, self.think_pattern, self.think_mode, self.think_remove_percent
        )
        result = self._process_for_block_type(
            result, self.tool_call_pattern, self.tool_call_mode, self.tool_call_remove_percent
        )
        return result
    
    def get_stats(
        self, original_messages: List[Dict[str, Any]], processed_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        original_chars = sum(len(m.get('content', '')) for m in original_messages)
        processed_chars = sum(len(m.get('content', '')) for m in processed_messages)
        
        original_think_count = sum(
            len(self.think_pattern.findall(m.get('content', '')))
            for m in original_messages if m.get('role') == 'assistant'
        )
        original_tool_call_count = sum(
            len(self.tool_call_pattern.findall(m.get('content', '')))
            for m in original_messages if m.get('role') == 'assistant'
        )
        processed_think_count = sum(
            len(self.think_pattern.findall(m.get('content', '')))
            for m in processed_messages if m.get('role') == 'assistant'
        )
        processed_tool_call_count = sum(
            len(self.tool_call_pattern.findall(m.get('content', '')))
            for m in processed_messages if m.get('role') == 'assistant'
        )
        
        return {
            'think_mode': self.think_mode.value,
            'tool_call_mode': self.tool_call_mode.value,
            'think_remove_percent': self.think_remove_percent if self.think_mode == ThinkRemovalMode.REMOVE_PERCENT else None,
            'tool_call_remove_percent': self.tool_call_remove_percent if self.tool_call_mode == ToolCallRemovalMode.REMOVE_PERCENT else None,
            'total_original_chars': original_chars,
            'total_processed_chars': processed_chars,
            'saved_chars': original_chars - processed_chars,
            'reduction_ratio': round((original_chars - processed_chars) / original_chars, 3) if original_chars > 0 else 0,
            'think_blocks': {
                'original': original_think_count,
                'processed': processed_think_count,
                'removed': original_think_count - processed_think_count
            },
            'tool_call_blocks': {
                'original': original_tool_call_count,
                'processed': processed_tool_call_count,
                'removed': original_tool_call_count - processed_tool_call_count
            }
        }


# ============================================================
# Part 2: Entity Knowledge Graph (MTM - Mid-Term Memory)
# ============================================================

class EntityType(Enum):
    """实体类型"""
    PAGE = "page"                    # 网页
    ELEMENT = "element"              # 页面元素
    DATA = "data"                    # 提取的数据
    FORM_FIELD = "form_field"        # 表单字段
    NAVIGATION = "navigation"        # 导航项
    ACTION_RESULT = "action_result"  # 动作结果
    ERROR = "error"                  # 错误信息
    USER_INPUT = "user_input"        # 用户输入的值


class RelationType(Enum):
    """关系类型"""
    CONTAINS = "contains"            # 包含关系
    LINKS_TO = "links_to"            # 链接到
    EXTRACTED_FROM = "extracted_from"  # 从...提取
    DEPENDS_ON = "depends_on"        # 依赖于
    LEADS_TO = "leads_to"            # 导致
    FAILED_WITH = "failed_with"      # 失败于
    SUCCEEDED_WITH = "succeeded_with"  # 成功于
    RELATED_TO = "related_to"        # 相关于
    PART_OF = "part_of"              # 是...的一部分
    TRIGGERS = "triggers"            # 触发


@dataclass
class Entity:
    """图中的实体节点"""
    id: str
    entity_type: EntityType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    relevance_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.entity_type.value,
            "name": self.name,
            "properties": self.properties,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "relevance_score": self.relevance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            id=data["id"],
            entity_type=EntityType(data["type"]),
            name=data["name"],
            properties=data.get("properties", {}),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            access_count=data.get("access_count", 0),
            relevance_score=data.get("relevance_score", 1.0)
        )
    
    def update(self, properties: Dict[str, Any]) -> None:
        self.properties.update(properties)
        self.updated_at = time.time()
        self.access_count += 1


@dataclass
class Relation:
    """图中的关系边"""
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.relation_type.value,
            "properties": self.properties,
            "weight": self.weight,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["type"]),
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0),
            created_at=data.get("created_at", time.time())
        )


class EntityKnowledgeGraph:
    """
    实体知识图谱 - MTM (中期记忆)
    
    存储和管理 Web 任务执行过程中发现的实体及其关系。
    支持图的增删改查、子图检索、以及基于相关性的剪枝。
    """
    
    def __init__(self, max_entities: int = 1000, decay_factor: float = 0.95):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)  # source -> targets
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)  # target -> sources
        self.max_entities = max_entities
        self.decay_factor = decay_factor
        self._entity_index_by_type: Dict[EntityType, Set[str]] = defaultdict(set)
        self._entity_index_by_name: Dict[str, Set[str]] = defaultdict(set)
    
    def _generate_entity_id(self, entity_type: EntityType, name: str, properties: Dict) -> str:
        """生成实体的唯一ID"""
        content = f"{entity_type.value}:{name}:{json.dumps(properties, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def add_entity(
        self, 
        entity_type: EntityType, 
        name: str, 
        properties: Optional[Dict[str, Any]] = None,
        entity_id: Optional[str] = None
    ) -> Entity:
        """
        添加实体到图中。如果实体已存在，则更新其属性。
        """
        properties = properties or {}
        
        if entity_id is None:
            entity_id = self._generate_entity_id(entity_type, name, properties)
        
        if entity_id in self.entities:
            # 更新现有实体
            existing = self.entities[entity_id]
            existing.update(properties)
            return existing
        
        # 检查是否需要剪枝
        if len(self.entities) >= self.max_entities:
            self._prune_least_relevant()
        
        # 创建新实体
        entity = Entity(
            id=entity_id,
            entity_type=entity_type,
            name=name,
            properties=properties
        )
        
        self.entities[entity_id] = entity
        self._entity_index_by_type[entity_type].add(entity_id)
        self._entity_index_by_name[name.lower()].add(entity_id)
        
        return entity
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        properties: Optional[Dict[str, Any]] = None,
        weight: float = 1.0
    ) -> Optional[Relation]:
        """
        添加关系到图中。如果源或目标实体不存在，返回 None。
        """
        if source_id not in self.entities or target_id not in self.entities:
            return None
        
        # 检查是否已存在相同关系
        for rel in self.relations:
            if (rel.source_id == source_id and 
                rel.target_id == target_id and 
                rel.relation_type == relation_type):
                # 更新现有关系
                if properties:
                    rel.properties.update(properties)
                rel.weight = max(rel.weight, weight)
                return rel
        
        relation = Relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
            weight=weight
        )
        
        self.relations.append(relation)
        self.adjacency[source_id].add(target_id)
        self.reverse_adjacency[target_id].add(source_id)
        
        return relation
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        entity = self.entities.get(entity_id)
        if entity:
            entity.access_count += 1
        return entity
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """按类型获取所有实体"""
        entity_ids = self._entity_index_by_type.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    def search_entities_by_name(self, name_query: str) -> List[Entity]:
        """按名称模糊搜索实体"""
        results = []
        query_lower = name_query.lower()
        for name, entity_ids in self._entity_index_by_name.items():
            if query_lower in name:
                for eid in entity_ids:
                    if eid in self.entities:
                        results.append(self.entities[eid])
        return results
    
    def get_neighbors(
        self, 
        entity_id: str, 
        hops: int = 1,
        direction: str = "both"  # "out", "in", "both"
    ) -> Set[str]:
        """
        获取实体的邻居节点
        
        Args:
            entity_id: 起始实体ID
            hops: 跳数
            direction: 方向 ("out"=出边, "in"=入边, "both"=双向)
        
        Returns:
            邻居实体ID集合
        """
        if entity_id not in self.entities:
            return set()
        
        visited = {entity_id}
        current_level = {entity_id}
        
        for _ in range(hops):
            next_level = set()
            for node in current_level:
                if direction in ("out", "both"):
                    next_level.update(self.adjacency.get(node, set()))
                if direction in ("in", "both"):
                    next_level.update(self.reverse_adjacency.get(node, set()))
            next_level -= visited
            visited.update(next_level)
            current_level = next_level
        
        visited.remove(entity_id)
        return visited
    
    def get_subgraph(
        self, 
        entity_ids: Set[str],
        include_relations: bool = True
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        获取包含指定实体的子图
        """
        entities = [self.entities[eid] for eid in entity_ids if eid in self.entities]
        
        if not include_relations:
            return entities, []
        
        relations = [
            rel for rel in self.relations
            if rel.source_id in entity_ids and rel.target_id in entity_ids
        ]
        
        return entities, relations
    
    def get_relevant_subgraph(
        self, 
        query_keywords: List[str],
        max_entities: int = 20,
        hops: int = 2
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        基于查询关键词获取相关子图 (RAG on Graph)
        
        Args:
            query_keywords: 查询关键词列表
            max_entities: 最大返回实体数
            hops: 扩展跳数
        
        Returns:
            相关实体和关系
        """
        # 1. 找到匹配的种子实体
        seed_entities = set()
        for keyword in query_keywords:
            matched = self.search_entities_by_name(keyword)
            for entity in matched:
                seed_entities.add(entity.id)
        
        # 2. 扩展到邻居
        expanded = set(seed_entities)
        for seed_id in seed_entities:
            neighbors = self.get_neighbors(seed_id, hops=hops, direction="both")
            expanded.update(neighbors)
        
        # 3. 按相关性评分排序并截断
        scored_entities = []
        for eid in expanded:
            if eid in self.entities:
                entity = self.entities[eid]
                # 计算分数: 基础相关性 + 是否为种子 + 访问频率
                score = entity.relevance_score
                if eid in seed_entities:
                    score += 2.0
                score += min(entity.access_count * 0.1, 1.0)
                scored_entities.append((score, eid))
        
        scored_entities.sort(reverse=True)
        selected_ids = {eid for _, eid in scored_entities[:max_entities]}
        
        return self.get_subgraph(selected_ids)
    
    def _prune_least_relevant(self, prune_count: int = 100) -> None:
        """移除最不相关的实体"""
        if len(self.entities) <= prune_count:
            return
        
        # 计算每个实体的综合分数
        scored = []
        for eid, entity in self.entities.items():
            # 时间衰减
            age = time.time() - entity.updated_at
            time_score = self.decay_factor ** (age / 3600)  # 每小时衰减
            
            # 综合分数
            score = entity.relevance_score * time_score + entity.access_count * 0.1
            scored.append((score, eid))
        
        scored.sort()
        
        # 移除分数最低的实体
        to_remove = [eid for _, eid in scored[:prune_count]]
        for eid in to_remove:
            self.remove_entity(eid)
    
    def remove_entity(self, entity_id: str) -> bool:
        """移除实体及其所有关系"""
        if entity_id not in self.entities:
            return False
        
        entity = self.entities[entity_id]
        
        # 从索引中移除
        self._entity_index_by_type[entity.entity_type].discard(entity_id)
        self._entity_index_by_name[entity.name.lower()].discard(entity_id)
        
        # 移除相关关系
        self.relations = [
            rel for rel in self.relations
            if rel.source_id != entity_id and rel.target_id != entity_id
        ]
        
        # 更新邻接表
        del self.adjacency[entity_id]
        del self.reverse_adjacency[entity_id]
        for neighbors in self.adjacency.values():
            neighbors.discard(entity_id)
        for neighbors in self.reverse_adjacency.values():
            neighbors.discard(entity_id)
        
        # 移除实体
        del self.entities[entity_id]
        
        return True
    
    def apply_delta(self, delta: Dict[str, Any]) -> None:
        """
        应用增量更新 (Delta Graph)
        
        Delta 格式:
        {
            "entities": [
                {"type": "page", "name": "...", "properties": {...}},
                ...
            ],
            "relations": [
                {"source": "entity_name", "target": "entity_name", "type": "contains", ...},
                ...
            ]
        }
        """
        entity_name_to_id: Dict[str, str] = {}
        
        # 1. 添加/更新实体
        for entity_data in delta.get("entities", []):
            entity_type = EntityType(entity_data.get("type", "data"))
            name = entity_data.get("name", "unnamed")
            properties = entity_data.get("properties", {})
            
            entity = self.add_entity(entity_type, name, properties)
            entity_name_to_id[name] = entity.id
        
        # 2. 添加关系
        for rel_data in delta.get("relations", []):
            source_name = rel_data.get("source")
            target_name = rel_data.get("target")
            rel_type = RelationType(rel_data.get("type", "related_to"))
            
            # 查找实体ID
            source_id = entity_name_to_id.get(source_name)
            target_id = entity_name_to_id.get(target_name)
            
            # 如果找不到，尝试搜索现有实体
            if not source_id:
                matches = self.search_entities_by_name(source_name)
                if matches:
                    source_id = matches[0].id
            if not target_id:
                matches = self.search_entities_by_name(target_name)
                if matches:
                    target_id = matches[0].id
            
            if source_id and target_id:
                self.add_relation(
                    source_id, target_id, rel_type,
                    properties=rel_data.get("properties", {}),
                    weight=rel_data.get("weight", 1.0)
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityKnowledgeGraph":
        """从字典反序列化"""
        graph = cls()
        for entity_data in data.get("entities", []):
            entity = Entity.from_dict(entity_data)
            graph.entities[entity.id] = entity
            graph._entity_index_by_type[entity.entity_type].add(entity.id)
            graph._entity_index_by_name[entity.name.lower()].add(entity.id)
        
        for rel_data in data.get("relations", []):
            rel = Relation.from_dict(rel_data)
            graph.relations.append(rel)
            graph.adjacency[rel.source_id].add(rel.target_id)
            graph.reverse_adjacency[rel.target_id].add(rel.source_id)
        
        return graph
    
    def to_natural_language(
        self, 
        entities: Optional[List[Entity]] = None,
        relations: Optional[List[Relation]] = None,
        max_length: int = 2000
    ) -> str:
        """
        将子图转换为自然语言描述，用于 Prompt
        """
        if entities is None:
            entities = list(self.entities.values())
        if relations is None:
            relations = self.relations
        
        lines = []
        
        # 按类型分组实体
        entities_by_type: Dict[EntityType, List[Entity]] = defaultdict(list)
        for e in entities:
            entities_by_type[e.entity_type].append(e)
        
        # 生成实体描述
        for entity_type, type_entities in entities_by_type.items():
            if type_entities:
                type_name = entity_type.value.replace("_", " ").title()
                lines.append(f"\n## {type_name}s:")
                for e in type_entities[:10]:  # 限制每类最多10个
                    prop_str = ", ".join(f"{k}={v}" for k, v in list(e.properties.items())[:5])
                    if prop_str:
                        lines.append(f"  - {e.name}: {prop_str}")
                    else:
                        lines.append(f"  - {e.name}")
        
        # 生成关系描述
        if relations:
            lines.append("\n## Relationships:")
            id_to_name = {e.id: e.name for e in entities}
            for rel in relations[:20]:  # 限制最多20个关系
                source_name = id_to_name.get(rel.source_id, rel.source_id)
                target_name = id_to_name.get(rel.target_id, rel.target_id)
                rel_type = rel.relation_type.value.replace("_", " ")
                lines.append(f"  - {source_name} --[{rel_type}]--> {target_name}")
        
        result = "\n".join(lines)
        
        # 截断
        if len(result) > max_length:
            result = result[:max_length - 20] + "\n... (truncated)"
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图的统计信息"""
        type_counts = defaultdict(int)
        for entity in self.entities.values():
            type_counts[entity.entity_type.value] += 1
        
        rel_type_counts = defaultdict(int)
        for rel in self.relations:
            rel_type_counts[rel.relation_type.value] += 1
        
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entities_by_type": dict(type_counts),
            "relations_by_type": dict(rel_type_counts)
        }


# ============================================================
# Part 3: Short-Term Memory (STM - Current Viewport)
# ============================================================

@dataclass
class ViewportInfo:
    """当前视口信息"""
    url: str = ""
    title: str = ""
    dom_summary: str = ""
    accessibility_tree: str = ""
    visible_elements: List[Dict[str, Any]] = field(default_factory=list)
    last_action: Optional[Dict[str, Any]] = None
    last_action_result: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "dom_summary": self.dom_summary,
            "accessibility_tree": self.accessibility_tree,
            "visible_elements": self.visible_elements,
            "last_action": self.last_action,
            "last_action_result": self.last_action_result,
            "error_message": self.error_message,
            "timestamp": self.timestamp
        }
    
    def to_prompt_text(self, max_length: int = 4000) -> str:
        """转换为 Prompt 文本"""
        lines = [
            f"## Current Page State",
            f"URL: {self.url}",
            f"Title: {self.title}",
        ]
        
        if self.last_action:
            action_str = json.dumps(self.last_action, ensure_ascii=False)
            lines.append(f"\nLast Action: {action_str}")
            if self.last_action_result:
                lines.append(f"Result: {self.last_action_result}")
        
        if self.error_message:
            lines.append(f"\n⚠️ Error: {self.error_message}")
        
        if self.accessibility_tree:
            lines.append(f"\n## Page Structure (Accessibility Tree):")
            # 截断过长的树
            tree_text = self.accessibility_tree
            if len(tree_text) > max_length // 2:
                tree_text = tree_text[:max_length // 2] + "\n... (truncated)"
            lines.append(tree_text)
        elif self.dom_summary:
            lines.append(f"\n## Page Summary:")
            lines.append(self.dom_summary)
        
        result = "\n".join(lines)
        if len(result) > max_length:
            result = result[:max_length - 20] + "\n... (truncated)"
        
        return result


class ShortTermMemory:
    """
    短期记忆 (STM) - 管理当前视口状态
    
    生命周期极短，页面跳转时自动重置。
    提供 Agent 执行操作的直接依据。
    """
    
    def __init__(self, history_size: int = 5):
        self.current: ViewportInfo = ViewportInfo()
        self.history: List[ViewportInfo] = []
        self.history_size = history_size
    
    def update(
        self,
        url: Optional[str] = None,
        title: Optional[str] = None,
        dom_summary: Optional[str] = None,
        accessibility_tree: Optional[str] = None,
        visible_elements: Optional[List[Dict[str, Any]]] = None,
        last_action: Optional[Dict[str, Any]] = None,
        last_action_result: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> None:
        """更新当前视口状态"""
        # 检测页面是否跳转
        if url and url != self.current.url and self.current.url:
            # 保存历史
            self.history.append(self.current)
            if len(self.history) > self.history_size:
                self.history.pop(0)
            # 创建新的视口
            self.current = ViewportInfo()
        
        # 更新当前状态
        if url is not None:
            self.current.url = url
        if title is not None:
            self.current.title = title
        if dom_summary is not None:
            self.current.dom_summary = dom_summary
        if accessibility_tree is not None:
            self.current.accessibility_tree = accessibility_tree
        if visible_elements is not None:
            self.current.visible_elements = visible_elements
        if last_action is not None:
            self.current.last_action = last_action
        if last_action_result is not None:
            self.current.last_action_result = last_action_result
        if error_message is not None:
            self.current.error_message = error_message
        
        self.current.timestamp = time.time()
    
    def get_current(self) -> ViewportInfo:
        """获取当前视口状态"""
        return self.current
    
    def get_navigation_history(self) -> List[str]:
        """获取导航历史 (URL 列表)"""
        urls = [v.url for v in self.history if v.url]
        if self.current.url:
            urls.append(self.current.url)
        return urls
    
    def clear(self) -> None:
        """清除所有短期记忆"""
        self.current = ViewportInfo()
        self.history.clear()
    
    def to_prompt_text(self, include_history: bool = False, max_length: int = 5000) -> str:
        """转换为 Prompt 文本"""
        text = self.current.to_prompt_text(max_length=max_length - 500 if include_history else max_length)
        
        if include_history and self.history:
            nav_history = self.get_navigation_history()[:-1]  # 不包括当前页面
            if nav_history:
                text += f"\n\n## Navigation History:\n"
                for i, url in enumerate(nav_history[-5:], 1):  # 最多显示5个
                    text += f"  {i}. {url}\n"
        
        return text


# ============================================================
# Part 4: Holographic Memory (Combined STM + MTM)
# ============================================================

class HolographicMemory:
    """
    全息记忆系统 - 整合 STM 和 MTM
    
    提供统一的记忆管理接口，包括：
    - 短期记忆 (STM): 当前视口状态
    - 中期记忆 (MTM): 实体知识图谱
    
    不包含长期记忆，因为 Web 任务不涉及跨会话的用户偏好。
    """
    
    def __init__(
        self,
        stm_history_size: int = 5,
        mtm_max_entities: int = 1000,
        mtm_decay_factor: float = 0.95
    ):
        self.stm = ShortTermMemory(history_size=stm_history_size)
        self.mtm = EntityKnowledgeGraph(
            max_entities=mtm_max_entities,
            decay_factor=mtm_decay_factor
        )
        self._message_processor = MemoryProcessor(
            think_mode=ThinkRemovalMode.REMOVE_ALL,
            tool_call_mode=ToolCallRemovalMode.KEEP_ALL
        )
    
    def update_viewport(self, **kwargs) -> None:
        """更新视口状态 (STM)"""
        self.stm.update(**kwargs)
        
        # 自动将页面信息添加到 MTM
        url = kwargs.get("url")
        title = kwargs.get("title")
        if url and title:
            self.mtm.add_entity(
                EntityType.PAGE,
                name=title or url,
                properties={"url": url, "title": title}
            )
    
    def apply_memory_delta(self, delta: Dict[str, Any]) -> None:
        """应用记忆增量更新 (MTM)"""
        self.mtm.apply_delta(delta)
    
    def get_relevant_context(
        self,
        keywords: List[str],
        max_entities: int = 15,
        hops: int = 2
    ) -> str:
        """
        获取与关键词相关的记忆上下文
        
        用于 RAG on Graph
        """
        entities, relations = self.mtm.get_relevant_subgraph(
            keywords, max_entities=max_entities, hops=hops
        )
        return self.mtm.to_natural_language(entities, relations)
    
    def get_full_context(self, max_stm_length: int = 4000, max_mtm_length: int = 2000) -> str:
        """获取完整的记忆上下文"""
        stm_text = self.stm.to_prompt_text(include_history=True, max_length=max_stm_length)
        
        # 获取最近/最相关的 MTM 实体
        recent_entities = sorted(
            self.mtm.entities.values(),
            key=lambda e: e.updated_at,
            reverse=True
        )[:20]
        
        if recent_entities:
            entity_ids = {e.id for e in recent_entities}
            _, relations = self.mtm.get_subgraph(entity_ids)
            mtm_text = self.mtm.to_natural_language(recent_entities, relations, max_length=max_mtm_length)
        else:
            mtm_text = ""
        
        sections = [stm_text]
        if mtm_text:
            sections.append("\n## Accumulated Knowledge:\n" + mtm_text)
        
        return "\n".join(sections)
    
    def process_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理消息历史 (移除 think 块等)"""
        return self._message_processor.process_messages(messages)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            "stm": {
                "current_url": self.stm.current.url,
                "history_length": len(self.stm.history)
            },
            "mtm": self.mtm.get_statistics()
        }
    
    def clear(self) -> None:
        """清除所有记忆"""
        self.stm.clear()
        self.mtm = EntityKnowledgeGraph(
            max_entities=self.mtm.max_entities,
            decay_factor=self.mtm.decay_factor
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "stm": self.stm.current.to_dict(),
            "stm_history": [v.to_dict() for v in self.stm.history],
            "mtm": self.mtm.to_dict()
        }


# ============================================================
# Part 5: Task DAG (Adaptive Planner)
# ============================================================

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"        # 待办
    ACTIVE = "active"          # 进行中
    COMPLETED = "completed"    # 完成
    FAILED = "failed"          # 失败
    BLOCKED = "blocked"        # 阻塞
    SKIPPED = "skipped"        # 跳过


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskNode:
    """任务节点"""
    id: str
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # 前置任务
    exit_conditions: List[str] = field(default_factory=list)  # 完成条件
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    attempt_count: int = 0
    max_attempts: int = 3
    error_message: Optional[str] = None
    result: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "dependencies": self.dependencies,
            "exit_conditions": self.exit_conditions,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "error_message": self.error_message,
            "result": self.result
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskNode":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            status=TaskStatus(data.get("status", "pending")),
            priority=TaskPriority(data.get("priority", 2)),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            dependencies=data.get("dependencies", []),
            exit_conditions=data.get("exit_conditions", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            attempt_count=data.get("attempt_count", 0),
            max_attempts=data.get("max_attempts", 3),
            error_message=data.get("error_message"),
            result=data.get("result")
        )
    
    def is_ready(self) -> bool:
        """检查任务是否可以开始"""
        return self.status == TaskStatus.PENDING
    
    def can_retry(self) -> bool:
        """检查任务是否可以重试"""
        return self.attempt_count < self.max_attempts


@dataclass
class ReflectionResult:
    """反思结果"""
    task_id: str
    is_completed: bool
    is_blocked: bool
    needs_replan: bool
    completion_reason: Optional[str] = None
    blockers: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    new_subtasks: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskDAG:
    """
    任务有向无环图 (Task DAG)
    
    支持非线性任务规划，包括：
    - 并行子任务
    - 条件分支
    - 动态重规划
    - 任务依赖管理
    """
    
    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}
        self.root_id: Optional[str] = None
        self._id_counter: int = 0
    
    def _generate_id(self) -> str:
        """生成唯一任务ID"""
        self._id_counter += 1
        return f"task_{self._id_counter}"
    
    def create_root_task(
        self,
        name: str,
        description: str,
        exit_conditions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskNode:
        """创建根任务"""
        task_id = self._generate_id()
        task = TaskNode(
            id=task_id,
            name=name,
            description=description,
            priority=TaskPriority.CRITICAL,
            exit_conditions=exit_conditions or [],
            metadata=metadata or {}
        )
        self.nodes[task_id] = task
        self.root_id = task_id
        return task
    
    def add_subtask(
        self,
        parent_id: str,
        name: str,
        description: str,
        dependencies: Optional[List[str]] = None,
        exit_conditions: Optional[List[str]] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[TaskNode]:
        """添加子任务"""
        if parent_id not in self.nodes:
            return None
        
        task_id = self._generate_id()
        task = TaskNode(
            id=task_id,
            name=name,
            description=description,
            parent_id=parent_id,
            dependencies=dependencies or [],
            exit_conditions=exit_conditions or [],
            priority=priority,
            metadata=metadata or {}
        )
        
        self.nodes[task_id] = task
        self.nodes[parent_id].children_ids.append(task_id)
        
        return task
    
    def add_parallel_subtasks(
        self,
        parent_id: str,
        tasks_info: List[Dict[str, Any]]
    ) -> List[TaskNode]:
        """添加并行子任务"""
        created = []
        for info in tasks_info:
            task = self.add_subtask(
                parent_id=parent_id,
                name=info.get("name", "Unnamed"),
                description=info.get("description", ""),
                dependencies=info.get("dependencies", []),
                exit_conditions=info.get("exit_conditions", []),
                priority=TaskPriority(info.get("priority", 2)),
                metadata=info.get("metadata", {})
            )
            if task:
                created.append(task)
        return created
    
    def get_task(self, task_id: str) -> Optional[TaskNode]:
        """获取任务"""
        return self.nodes.get(task_id)
    
    def get_active_tasks(self) -> List[TaskNode]:
        """获取所有活动任务"""
        return [t for t in self.nodes.values() if t.status == TaskStatus.ACTIVE]
    
    def get_pending_tasks(self) -> List[TaskNode]:
        """获取所有待办任务"""
        return [t for t in self.nodes.values() if t.status == TaskStatus.PENDING]
    
    def get_ready_tasks(self) -> List[TaskNode]:
        """
        获取所有可以开始的任务
        (状态为 PENDING 且所有依赖已完成)
        """
        ready = []
        for task in self.get_pending_tasks():
            deps_satisfied = all(
                self.nodes.get(dep_id, TaskNode(id="", name="", description="")).status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            if deps_satisfied:
                ready.append(task)
        
        # 按优先级排序
        ready.sort(key=lambda t: t.priority.value, reverse=True)
        return ready
    
    def get_next_task(self) -> Optional[TaskNode]:
        """获取下一个应该执行的任务"""
        # 1. 优先返回活动任务
        active = self.get_active_tasks()
        if active:
            return active[0]
        
        # 2. 否则返回优先级最高的就绪任务
        ready = self.get_ready_tasks()
        if ready:
            return ready[0]
        
        return None
    
    def start_task(self, task_id: str) -> bool:
        """开始任务"""
        task = self.nodes.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False
        
        task.status = TaskStatus.ACTIVE
        task.started_at = time.time()
        task.attempt_count += 1
        return True
    
    def complete_task(
        self,
        task_id: str,
        result: Any = None,
        reason: Optional[str] = None
    ) -> bool:
        """完成任务"""
        task = self.nodes.get(task_id)
        if not task:
            return False
        
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.result = result
        if reason:
            task.metadata["completion_reason"] = reason
        
        # 检查父任务是否所有子任务都已完成
        self._check_parent_completion(task_id)
        
        return True
    
    def fail_task(
        self,
        task_id: str,
        error_message: str,
        can_retry: bool = True
    ) -> bool:
        """标记任务失败"""
        task = self.nodes.get(task_id)
        if not task:
            return False
        
        task.error_message = error_message
        
        if can_retry and task.can_retry():
            task.status = TaskStatus.PENDING  # 重置为待办以便重试
        else:
            task.status = TaskStatus.FAILED
            # 阻塞依赖此任务的其他任务
            self._propagate_failure(task_id)
        
        return True
    
    def block_task(self, task_id: str, blockers: List[str]) -> bool:
        """阻塞任务"""
        task = self.nodes.get(task_id)
        if not task:
            return False
        
        task.status = TaskStatus.BLOCKED
        task.metadata["blockers"] = blockers
        return True
    
    def skip_task(self, task_id: str, reason: str) -> bool:
        """跳过任务"""
        task = self.nodes.get(task_id)
        if not task:
            return False
        
        task.status = TaskStatus.SKIPPED
        task.metadata["skip_reason"] = reason
        return True
    
    def _check_parent_completion(self, task_id: str) -> None:
        """检查父任务是否可以标记为完成"""
        task = self.nodes.get(task_id)
        if not task or not task.parent_id:
            return
        
        parent = self.nodes.get(task.parent_id)
        if not parent:
            return
        
        # 检查所有子任务状态
        children = [self.nodes.get(cid) for cid in parent.children_ids]
        children = [c for c in children if c is not None]
        
        all_done = all(
            c.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
            for c in children
        )
        
        if all_done and parent.status == TaskStatus.ACTIVE:
            self.complete_task(parent.id, reason="All subtasks completed")
    
    def _propagate_failure(self, task_id: str) -> None:
        """传播失败状态到依赖任务"""
        for task in self.nodes.values():
            if task_id in task.dependencies and task.status == TaskStatus.PENDING:
                self.block_task(task.id, [f"Dependency {task_id} failed"])
    
    def insert_subtask_before(
        self,
        target_task_id: str,
        name: str,
        description: str,
        **kwargs
    ) -> Optional[TaskNode]:
        """在目标任务之前插入新子任务"""
        target = self.nodes.get(target_task_id)
        if not target or not target.parent_id:
            return None
        
        # 新任务继承目标任务的依赖
        new_task = self.add_subtask(
            parent_id=target.parent_id,
            name=name,
            description=description,
            dependencies=target.dependencies.copy(),
            **kwargs
        )
        
        if new_task:
            # 目标任务现在依赖新任务
            target.dependencies = [new_task.id]
        
        return new_task
    
    def replan_from_task(
        self,
        task_id: str,
        new_subtasks: List[Dict[str, Any]]
    ) -> List[TaskNode]:
        """从某个任务点开始重规划"""
        task = self.nodes.get(task_id)
        if not task:
            return []
        
        # 取消原有的未完成子任务
        for child_id in task.children_ids:
            child = self.nodes.get(child_id)
            if child and child.status in (TaskStatus.PENDING, TaskStatus.BLOCKED):
                self.skip_task(child_id, "Replaced by replan")
        
        # 添加新子任务
        return self.add_parallel_subtasks(task_id, new_subtasks)
    
    def get_task_path(self, task_id: str) -> List[TaskNode]:
        """获取从根到指定任务的路径"""
        path = []
        current_id = task_id
        
        while current_id:
            task = self.nodes.get(current_id)
            if not task:
                break
            path.append(task)
            current_id = task.parent_id
        
        path.reverse()
        return path
    
    def get_progress(self) -> Dict[str, Any]:
        """获取任务进度统计"""
        total = len(self.nodes)
        by_status = defaultdict(int)
        for task in self.nodes.values():
            by_status[task.status.value] += 1
        
        completed = by_status.get("completed", 0) + by_status.get("skipped", 0)
        
        return {
            "total_tasks": total,
            "completed": completed,
            "pending": by_status.get("pending", 0),
            "active": by_status.get("active", 0),
            "failed": by_status.get("failed", 0),
            "blocked": by_status.get("blocked", 0),
            "progress_percent": round(completed / total * 100, 1) if total > 0 else 0,
            "by_status": dict(by_status)
        }
    
    def to_prompt_text(self, max_depth: int = 3) -> str:
        """转换为 Prompt 文本"""
        if not self.root_id:
            return "No task plan defined."
        
        lines = ["## Task Plan:"]
        
        def format_task(task: TaskNode, depth: int, prefix: str = "") -> None:
            if depth > max_depth:
                return
            
            status_emoji = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.ACTIVE: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.BLOCKED: "🚫",
                TaskStatus.SKIPPED: "⏭️"
            }
            
            emoji = status_emoji.get(task.status, "•")
            indent = "  " * depth
            
            lines.append(f"{indent}{prefix}{emoji} {task.name} [{task.status.value}]")
            if task.description and depth < 2:
                lines.append(f"{indent}   └─ {task.description[:100]}")
            
            for i, child_id in enumerate(task.children_ids):
                child = self.nodes.get(child_id)
                if child:
                    is_last = i == len(task.children_ids) - 1
                    child_prefix = "└─ " if is_last else "├─ "
                    format_task(child, depth + 1, child_prefix)
        
        root = self.nodes.get(self.root_id)
        if root:
            format_task(root, 0)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "root_id": self.root_id,
            "nodes": {tid: t.to_dict() for tid, t in self.nodes.items()},
            "_id_counter": self._id_counter
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskDAG":
        """反序列化"""
        dag = cls()
        dag.root_id = data.get("root_id")
        dag._id_counter = data.get("_id_counter", 0)
        for tid, tdata in data.get("nodes", {}).items():
            dag.nodes[tid] = TaskNode.from_dict(tdata)
        return dag


# ============================================================
# Part 6: Adaptive Planner (Plan-Execute-Reflect-Memorize)
# ============================================================

class AdaptivePlanner:
    """
    自适应规划器
    
    实现 Plan-Execute-Reflect-Memorize 循环：
    1. Plan: 分解任务为 DAG
    2. Execute: 执行当前子任务
    3. Reflect: 反思执行结果
    4. Memorize: 更新记忆图
    """
    
    def __init__(
        self,
        memory: HolographicMemory,
        max_retries: int = 3,
        stuck_threshold: int = 5
    ):
        self.memory = memory
        self.task_dag = TaskDAG()
        self.max_retries = max_retries
        self.stuck_threshold = stuck_threshold
        self._step_count = 0
        self._no_progress_count = 0
        self._last_progress_snapshot: Optional[Dict] = None
    
    def initialize_task(
        self,
        task_name: str,
        task_description: str,
        initial_subtasks: Optional[List[Dict[str, Any]]] = None,
        exit_conditions: Optional[List[str]] = None
    ) -> TaskNode:
        """
        初始化主任务
        
        Args:
            task_name: 任务名称
            task_description: 任务描述
            initial_subtasks: 初始子任务列表
            exit_conditions: 完成条件
        """
        root = self.task_dag.create_root_task(
            name=task_name,
            description=task_description,
            exit_conditions=exit_conditions or []
        )
        
        # 添加到记忆
        self.memory.mtm.add_entity(
            EntityType.DATA,
            name=f"Task: {task_name}",
            properties={
                "type": "root_task",
                "description": task_description,
                "task_id": root.id
            }
        )
        
        # 添加初始子任务
        if initial_subtasks:
            self.task_dag.add_parallel_subtasks(root.id, initial_subtasks)
        
        # 启动根任务
        self.task_dag.start_task(root.id)
        
        return root
    
    def get_current_task(self) -> Optional[TaskNode]:
        """获取当前应该执行的任务"""
        return self.task_dag.get_next_task()
    
    def execute_step(self) -> Optional[TaskNode]:
        """
        执行一步
        
        Returns:
            当前活动的任务节点，如果没有则返回 None
        """
        self._step_count += 1
        
        current = self.get_current_task()
        if not current:
            return None
        
        # 如果是 PENDING 状态，激活它
        if current.status == TaskStatus.PENDING:
            self.task_dag.start_task(current.id)
        
        return current
    
    def reflect(
        self,
        task_id: str,
        action_result: Optional[str] = None,
        error: Optional[str] = None,
        observed_data: Optional[Dict[str, Any]] = None
    ) -> ReflectionResult:
        """
        反思执行结果
        
        检查：
        1. 当前子任务是否完成
        2. 是否遇到阻塞
        3. 是否需要重新规划
        """
        task = self.task_dag.get_task(task_id)
        if not task:
            return ReflectionResult(
                task_id=task_id,
                is_completed=False,
                is_blocked=True,
                needs_replan=False,
                blockers=["Task not found"]
            )
        
        result = ReflectionResult(
            task_id=task_id,
            is_completed=False,
            is_blocked=False,
            needs_replan=False
        )
        
        # 1. 检查是否有错误
        if error:
            result.blockers.append(error)
            if task.can_retry():
                result.suggested_actions.append(f"Retry task (attempt {task.attempt_count + 1}/{task.max_attempts})")
            else:
                result.is_blocked = True
                result.needs_replan = True
                result.suggested_actions.append("Find alternative approach")
        
        # 2. 检查退出条件
        if task.exit_conditions and observed_data:
            conditions_met = self._check_exit_conditions(task.exit_conditions, observed_data)
            if conditions_met:
                result.is_completed = True
                result.completion_reason = "Exit conditions satisfied"
        
        # 3. 检查是否卡住
        progress = self.task_dag.get_progress()
        if self._is_stuck(progress):
            result.needs_replan = True
            result.suggested_actions.append("Re-evaluate approach due to lack of progress")
        
        # 4. 记录反思结果到记忆
        if observed_data:
            self._memorize_observations(task, observed_data)
        
        return result
    
    def apply_reflection(self, reflection: ReflectionResult) -> None:
        """应用反思结果"""
        task = self.task_dag.get_task(reflection.task_id)
        if not task:
            return
        
        if reflection.is_completed:
            self.task_dag.complete_task(
                reflection.task_id,
                reason=reflection.completion_reason
            )
        elif reflection.is_blocked:
            self.task_dag.block_task(reflection.task_id, reflection.blockers)
        
        # 添加新子任务
        if reflection.new_subtasks:
            self.task_dag.add_parallel_subtasks(reflection.task_id, reflection.new_subtasks)
    
    def replan(
        self,
        from_task_id: str,
        new_plan: List[Dict[str, Any]],
        reason: str
    ) -> List[TaskNode]:
        """
        从某个任务点开始重规划
        
        Args:
            from_task_id: 起始任务ID
            new_plan: 新的子任务计划
            reason: 重规划原因
        """
        # 记录重规划事件
        self.memory.mtm.add_entity(
            EntityType.ACTION_RESULT,
            name=f"Replan at {from_task_id}",
            properties={
                "reason": reason,
                "new_tasks_count": len(new_plan),
                "timestamp": time.time()
            }
        )
        
        return self.task_dag.replan_from_task(from_task_id, new_plan)
    
    def _check_exit_conditions(
        self,
        conditions: List[str],
        observed_data: Dict[str, Any]
    ) -> bool:
        """检查退出条件是否满足"""
        # 简单实现：检查 observed_data 中是否包含条件关键词
        for condition in conditions:
            condition_lower = condition.lower()
            for key, value in observed_data.items():
                if condition_lower in str(key).lower() or condition_lower in str(value).lower():
                    return True
        return False
    
    def _is_stuck(self, progress: Dict[str, Any]) -> bool:
        """检查是否卡住（无进展）"""
        if self._last_progress_snapshot is None:
            self._last_progress_snapshot = progress
            return False
        
        # 比较进度
        if progress["completed"] == self._last_progress_snapshot.get("completed", 0):
            self._no_progress_count += 1
        else:
            self._no_progress_count = 0
        
        self._last_progress_snapshot = progress
        
        return self._no_progress_count >= self.stuck_threshold
    
    def _memorize_observations(
        self,
        task: TaskNode,
        observed_data: Dict[str, Any]
    ) -> None:
        """将观察到的数据添加到记忆"""
        for key, value in observed_data.items():
            if isinstance(value, (str, int, float, bool)):
                entity = self.memory.mtm.add_entity(
                    EntityType.DATA,
                    name=key,
                    properties={
                        "value": value,
                        "source_task": task.id,
                        "task_name": task.name
                    }
                )
                
                # 建立与任务的关系
                task_entity_matches = self.memory.mtm.search_entities_by_name(f"Task: {task.name}")
                if task_entity_matches:
                    self.memory.mtm.add_relation(
                        task_entity_matches[0].id,
                        entity.id,
                        RelationType.EXTRACTED_FROM
                    )
    
    def get_context_for_prompt(
        self,
        include_task_tree: bool = True,
        include_memory: bool = True,
        max_context_length: int = 6000
    ) -> str:
        """
        获取用于 Prompt 的上下文
        
        整合：
        1. 当前任务信息
        2. 任务树
        3. 相关记忆
        4. 当前视口状态
        """
        sections = []
        
        # 1. 当前任务
        current = self.get_current_task()
        if current:
            task_path = self.task_dag.get_task_path(current.id)
            path_str = " > ".join(t.name for t in task_path)
            
            sections.append(f"""## Current Task
Path: {path_str}
Name: {current.name}
Description: {current.description}
Status: {current.status.value}
Attempt: {current.attempt_count}/{current.max_attempts}""")
            
            if current.exit_conditions:
                sections.append(f"Exit Conditions: {', '.join(current.exit_conditions)}")
        
        # 2. 任务树
        if include_task_tree:
            tree_text = self.task_dag.to_prompt_text(max_depth=2)
            sections.append(tree_text)
        
        # 3. 进度
        progress = self.task_dag.get_progress()
        sections.append(f"""## Progress
Completed: {progress['completed']}/{progress['total_tasks']} ({progress['progress_percent']}%)""")
        
        # 4. 记忆上下文
        if include_memory:
            # 根据当前任务提取相关关键词
            keywords = []
            if current:
                keywords.extend(current.name.split())
                keywords.extend(current.description.split()[:10])
            
            if keywords:
                memory_context = self.memory.get_relevant_context(
                    keywords=keywords,
                    max_entities=10,
                    hops=1
                )
                if memory_context.strip():
                    sections.append(f"## Relevant Knowledge\n{memory_context}")
        
        # 5. 当前视口
        viewport_text = self.memory.stm.to_prompt_text(include_history=False, max_length=2000)
        sections.append(viewport_text)
        
        result = "\n\n".join(sections)
        
        # 截断
        if len(result) > max_context_length:
            result = result[:max_context_length - 50] + "\n\n... (context truncated)"
        
        return result
    
    def is_complete(self) -> bool:
        """检查整个任务是否完成"""
        if not self.task_dag.root_id:
            return True
        
        root = self.task_dag.get_task(self.task_dag.root_id)
        return root is not None and root.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
    
    def get_final_result(self) -> Dict[str, Any]:
        """获取最终结果"""
        progress = self.task_dag.get_progress()
        
        result = {
            "success": progress["failed"] == 0 and progress["blocked"] == 0,
            "progress": progress,
            "total_steps": self._step_count,
            "memory_stats": self.memory.get_statistics()
        }
        
        # 收集所有完成任务的结果
        results = {}
        for task in self.task_dag.nodes.values():
            if task.status == TaskStatus.COMPLETED and task.result:
                results[task.name] = task.result
        
        result["task_results"] = results
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "task_dag": self.task_dag.to_dict(),
            "memory": self.memory.to_dict(),
            "step_count": self._step_count,
            "no_progress_count": self._no_progress_count
        }


# ============================================================
# Part 7: Context Synthesizer
# ============================================================

class ContextSynthesizer:
    """
    上下文合成器
    
    在每次 LLM 推理前，将记忆和规划信息合成为统一的 Prompt。
    """
    
    def __init__(
        self,
        planner: AdaptivePlanner,
        max_context_tokens: int = 8000,
        include_system_prompt: bool = True
    ):
        self.planner = planner
        self.max_context_tokens = max_context_tokens
        self.include_system_prompt = include_system_prompt
        self._char_to_token_ratio = 4  # 粗略估计
    
    def _estimate_tokens(self, text: str) -> int:
        """估计文本的 token 数"""
        return len(text) // self._char_to_token_ratio
    
    def synthesize(
        self,
        user_query: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        output_format: Optional[str] = None
    ) -> str:
        """
        合成完整的上下文 Prompt
        
        包含：
        1. 系统指令
        2. 任务上下文
        3. 当前状态
        4. 用户查询
        5. 输出格式要求
        """
        sections = []
        
        # 1. 获取规划器上下文
        planner_context = self.planner.get_context_for_prompt(
            include_task_tree=True,
            include_memory=True,
            max_context_length=self.max_context_tokens * self._char_to_token_ratio // 2
        )
        sections.append(planner_context)
        
        # 2. 添加额外指令
        if additional_instructions:
            sections.append(f"## Additional Instructions\n{additional_instructions}")
        
        # 3. 添加用户查询
        if user_query:
            sections.append(f"## User Query\n{user_query}")
        
        # 4. 添加输出格式
        if output_format:
            sections.append(f"## Expected Output Format\n{output_format}")
        
        return "\n\n".join(sections)
    
    def create_action_prompt(
        self,
        available_actions: List[Dict[str, Any]],
        constraints: Optional[List[str]] = None
    ) -> str:
        """创建动作选择 Prompt"""
        context = self.synthesize()
        
        # 添加可用动作
        actions_text = "## Available Actions\n"
        for action in available_actions:
            actions_text += f"- **{action['name']}**: {action.get('description', '')}\n"
            if action.get('parameters'):
                actions_text += f"  Parameters: {json.dumps(action['parameters'])}\n"
        
        context += f"\n\n{actions_text}"
        
        # 添加约束
        if constraints:
            context += "\n\n## Constraints\n"
            for c in constraints:
                context += f"- {c}\n"
        
        # 添加输出格式要求
        context += """
## Output Format
Provide your response in the following format:

<think>
[Your reasoning about what action to take and why]
</think>

<tool_call>
{"name": "action_name", "arguments": {...}}
</tool_call>

<memory_delta>
{"entities": [...], "relations": [...]}
</memory_delta>

<reflection>
{"task_status": "in_progress|completed|blocked", "observations": {...}, "needs_replan": false}
</reflection>
"""
        
        return context
    
    def parse_agent_output(self, output: str) -> Dict[str, Any]:
        """解析 Agent 输出"""
        result = {
            "think": None,
            "tool_call": None,
            "memory_delta": None,
            "reflection": None,
            "answer": None,
            "raw": output
        }
        
        # 解析 think
        think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
        if think_match:
            result["think"] = think_match.group(1).strip()
        
        # 解析 tool_call
        tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', output, re.DOTALL)
        if tool_call_match:
            try:
                result["tool_call"] = json.loads(tool_call_match.group(1).strip())
            except json.JSONDecodeError:
                result["tool_call"] = tool_call_match.group(1).strip()
        
        # 解析 memory_delta
        memory_match = re.search(r'<memory_delta>(.*?)</memory_delta>', output, re.DOTALL)
        if memory_match:
            try:
                result["memory_delta"] = json.loads(memory_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # 解析 reflection
        reflection_match = re.search(r'<reflection>(.*?)</reflection>', output, re.DOTALL)
        if reflection_match:
            try:
                result["reflection"] = json.loads(reflection_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # 解析 answer
        answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
        if answer_match:
            result["answer"] = answer_match.group(1).strip()
        
        return result


# ============================================================
# Part 8: Web Agent Controller (Integration)
# ============================================================

class WebAgentController:
    """
    Web Agent 控制器
    
    整合所有组件，提供统一的执行接口：
    - 初始化任务
    - 执行步骤
    - 处理响应
    - 管理记忆和规划
    """
    
    def __init__(
        self,
        max_steps: int = 50,
        stm_history_size: int = 5,
        mtm_max_entities: int = 500,
        stuck_threshold: int = 5
    ):
        self.max_steps = max_steps
        
        # 初始化记忆系统
        self.memory = HolographicMemory(
            stm_history_size=stm_history_size,
            mtm_max_entities=mtm_max_entities
        )
        
        # 初始化规划器
        self.planner = AdaptivePlanner(
            memory=self.memory,
            stuck_threshold=stuck_threshold
        )
        
        # 初始化上下文合成器
        self.synthesizer = ContextSynthesizer(planner=self.planner)
        
        self._step_count = 0
        self._is_initialized = False
    
    def initialize(
        self,
        task_name: str,
        task_description: str,
        initial_url: Optional[str] = None,
        initial_subtasks: Optional[List[Dict[str, Any]]] = None,
        exit_conditions: Optional[List[str]] = None
    ) -> None:
        """初始化 Agent"""
        # 初始化任务
        self.planner.initialize_task(
            task_name=task_name,
            task_description=task_description,
            initial_subtasks=initial_subtasks,
            exit_conditions=exit_conditions
        )
        
        # 初始化视口
        if initial_url:
            self.memory.update_viewport(url=initial_url)
        
        self._is_initialized = True
    
    def step(
        self,
        viewport_update: Optional[Dict[str, Any]] = None,
        available_actions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        执行一步
        
        Returns:
            包含 prompt, current_task, 等信息的字典
        """
        if not self._is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        self._step_count += 1
        
        if self._step_count > self.max_steps:
            return {
                "status": "max_steps_reached",
                "prompt": None,
                "current_task": None
            }
        
        # 更新视口
        if viewport_update:
            self.memory.update_viewport(**viewport_update)
        
        # 获取当前任务
        current_task = self.planner.execute_step()
        
        if not current_task:
            return {
                "status": "no_task",
                "prompt": None,
                "current_task": None
            }
        
        # 生成 Prompt
        if available_actions:
            prompt = self.synthesizer.create_action_prompt(available_actions)
        else:
            prompt = self.synthesizer.synthesize()
        
        return {
            "status": "ready",
            "prompt": prompt,
            "current_task": current_task.to_dict(),
            "step": self._step_count
        }
    
    def process_response(
        self,
        agent_output: str,
        action_result: Optional[str] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理 Agent 响应
        
        执行 Reflect 和 Memorize 阶段
        """
        # 解析输出
        parsed = self.synthesizer.parse_agent_output(agent_output)
        
        current_task = self.planner.get_current_task()
        if not current_task:
            return {"status": "no_task", "parsed": parsed}
        
        # 应用记忆增量
        if parsed.get("memory_delta"):
            self.memory.apply_memory_delta(parsed["memory_delta"])
        
        # 进行反思
        observed_data = {}
        if parsed.get("reflection"):
            observed_data = parsed["reflection"].get("observations", {})
        
        reflection = self.planner.reflect(
            task_id=current_task.id,
            action_result=action_result,
            error=error,
            observed_data=observed_data
        )
        
        # 检查是否需要重规划
        if parsed.get("reflection", {}).get("needs_replan"):
            reflection.needs_replan = True
        
        # 应用反思结果
        self.planner.apply_reflection(reflection)
        
        return {
            "status": "processed",
            "parsed": parsed,
            "reflection": {
                "is_completed": reflection.is_completed,
                "is_blocked": reflection.is_blocked,
                "needs_replan": reflection.needs_replan,
                "blockers": reflection.blockers
            },
            "is_complete": self.planner.is_complete()
        }
    
    def get_final_result(self) -> Dict[str, Any]:
        """获取最终结果"""
        return self.planner.get_final_result()
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            "initialized": self._is_initialized,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "is_complete": self.planner.is_complete() if self._is_initialized else False,
            "progress": self.planner.task_dag.get_progress() if self._is_initialized else None,
            "memory_stats": self.memory.get_statistics()
        }


# ============================================================
# Factory Functions and Convenience Methods
# ============================================================

def create_memory_processor(
    think_mode: str = "remove_all",
    tool_call_mode: str = "keep_all",
    think_remove_percent: float = 1.0,
    tool_call_remove_percent: float = 1.0
) -> MemoryProcessor:
    """工厂函数，创建 MemoryProcessor 实例"""
    think_mode_map = {
        "remove_all": ThinkRemovalMode.REMOVE_ALL,
        "keep_last": ThinkRemovalMode.KEEP_LAST,
        "keep_all": ThinkRemovalMode.KEEP_ALL,
        "remove_percent": ThinkRemovalMode.REMOVE_PERCENT,
    }
    
    tool_call_mode_map = {
        "remove_all": ToolCallRemovalMode.REMOVE_ALL,
        "keep_last": ToolCallRemovalMode.KEEP_LAST,
        "keep_all": ToolCallRemovalMode.KEEP_ALL,
        "remove_percent": ToolCallRemovalMode.REMOVE_PERCENT,
    }
    
    return MemoryProcessor(
        think_mode=think_mode_map.get(think_mode, ThinkRemovalMode.REMOVE_ALL),
        tool_call_mode=tool_call_mode_map.get(tool_call_mode, ToolCallRemovalMode.KEEP_ALL),
        think_remove_percent=think_remove_percent,
        tool_call_remove_percent=tool_call_remove_percent
    )


def create_web_agent(
    task_name: str,
    task_description: str,
    initial_url: Optional[str] = None,
    initial_subtasks: Optional[List[Dict[str, Any]]] = None,
    max_steps: int = 50
) -> WebAgentController:
    """
    便捷函数：创建并初始化 Web Agent
    """
    agent = WebAgentController(max_steps=max_steps)
    agent.initialize(
        task_name=task_name,
        task_description=task_description,
        initial_url=initial_url,
        initial_subtasks=initial_subtasks
    )
    return agent


# Convenience functions for backward compatibility
def process_messages_remove_all_think(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processor = MemoryProcessor(think_mode=ThinkRemovalMode.REMOVE_ALL, tool_call_mode=ToolCallRemovalMode.KEEP_ALL)
    return processor.process_messages(messages)


def process_messages_keep_last_think(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processor = MemoryProcessor(think_mode=ThinkRemovalMode.KEEP_LAST, tool_call_mode=ToolCallRemovalMode.KEEP_ALL)
    return processor.process_messages(messages)


def process_messages_remove_percent_think(messages: List[Dict[str, Any]], percent: float = 0.75) -> List[Dict[str, Any]]:
    processor = MemoryProcessor(think_mode=ThinkRemovalMode.REMOVE_PERCENT, tool_call_mode=ToolCallRemovalMode.KEEP_ALL, think_remove_percent=percent)
    return processor.process_messages(messages)


def process_messages_remove_all_tool_call(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processor = MemoryProcessor(think_mode=ThinkRemovalMode.KEEP_ALL, tool_call_mode=ToolCallRemovalMode.REMOVE_ALL)
    return processor.process_messages(messages)


def process_messages_keep_last_tool_call(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processor = MemoryProcessor(think_mode=ThinkRemovalMode.KEEP_ALL, tool_call_mode=ToolCallRemovalMode.KEEP_LAST)
    return processor.process_messages(messages)


def process_messages_remove_percent_tool_call(messages: List[Dict[str, Any]], percent: float = 0.75) -> List[Dict[str, Any]]:
    processor = MemoryProcessor(think_mode=ThinkRemovalMode.KEEP_ALL, tool_call_mode=ToolCallRemovalMode.REMOVE_PERCENT, tool_call_remove_percent=percent)
    return processor.process_messages(messages)


def process_messages_remove_all(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return process_messages_remove_all_think(messages)


def process_messages_keep_last(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return process_messages_keep_last_think(messages)


def process_messages_remove_all_think_and_tool_call(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processor = MemoryProcessor(think_mode=ThinkRemovalMode.REMOVE_ALL, tool_call_mode=ToolCallRemovalMode.REMOVE_ALL)
    return processor.process_messages(messages)


def process_messages_keep_last_think_and_tool_call(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processor = MemoryProcessor(think_mode=ThinkRemovalMode.KEEP_LAST, tool_call_mode=ToolCallRemovalMode.KEEP_LAST)
    return processor.process_messages(messages)


def process_messages_remove_percent_think_and_tool_call(
    messages: List[Dict[str, Any]],
    think_percent: float = 0.75,
    tool_call_percent: float = 0.75
) -> List[Dict[str, Any]]:
    processor = MemoryProcessor(
        think_mode=ThinkRemovalMode.REMOVE_PERCENT,
        tool_call_mode=ToolCallRemovalMode.REMOVE_PERCENT,
        think_remove_percent=think_percent,
        tool_call_remove_percent=tool_call_percent
    )
    return processor.process_messages(messages)


# ============================================================
# Test Code
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Testing Web Agent Memory System")
    print("=" * 80)
    
    # 1. 测试实体知识图谱
    print("\n--- Testing Entity Knowledge Graph (MTM) ---")
    graph = EntityKnowledgeGraph()
    
    # 添加实体
    page1 = graph.add_entity(EntityType.PAGE, "Google Search", {"url": "https://google.com"})
    page2 = graph.add_entity(EntityType.PAGE, "USGS Species Database", {"url": "https://nas.er.usgs.gov"})
    data1 = graph.add_entity(EntityType.DATA, "Clownfish Info", {"species": "Amphiprion ocellaris", "count": 5})
    
    # 添加关系
    graph.add_relation(page1.id, page2.id, RelationType.LINKS_TO)
    graph.add_relation(page2.id, data1.id, RelationType.CONTAINS)
    
    print(f"Graph stats: {graph.get_statistics()}")
    print(f"Natural language:\n{graph.to_natural_language()}")
    
    # 2. 测试短期记忆
    print("\n--- Testing Short-Term Memory (STM) ---")
    stm = ShortTermMemory()
    stm.update(url="https://google.com", title="Google", dom_summary="Search page")
    stm.update(url="https://nas.er.usgs.gov", title="NAS Database", accessibility_tree="[tree content]")
    print(f"Navigation history: {stm.get_navigation_history()}")
    print(f"Current viewport:\n{stm.to_prompt_text()[:500]}...")
    
    # 3. 测试全息记忆
    print("\n--- Testing Holographic Memory ---")
    memory = HolographicMemory()
    memory.update_viewport(url="https://google.com", title="Google")
    memory.apply_memory_delta({
        "entities": [
            {"type": "data", "name": "Search Result", "properties": {"query": "invasive species"}}
        ]
    })
    print(f"Memory stats: {memory.get_statistics()}")
    
    # 4. 测试任务 DAG
    print("\n--- Testing Task DAG ---")
    dag = TaskDAG()
    root = dag.create_root_task("Find Invasive Species", "Search for invasive species in Florida")
    
    sub1 = dag.add_subtask(root.id, "Search Google", "Search for species database")
    sub2 = dag.add_subtask(root.id, "Visit USGS", "Navigate to USGS database", dependencies=[sub1.id])
    sub3 = dag.add_subtask(root.id, "Extract Data", "Get species information", dependencies=[sub2.id])
    
    dag.start_task(root.id)
    dag.start_task(sub1.id)
    dag.complete_task(sub1.id, result={"found": True})
    
    print(dag.to_prompt_text())
    print(f"Progress: {dag.get_progress()}")
    
    # 5. 测试自适应规划器
    print("\n--- Testing Adaptive Planner ---")
    planner = AdaptivePlanner(memory=memory)
    planner.initialize_task(
        task_name="Research Task",
        task_description="Find information about invasive aquatic species",
        initial_subtasks=[
            {"name": "Search", "description": "Search for databases"},
            {"name": "Navigate", "description": "Go to USGS site"},
            {"name": "Extract", "description": "Get species data"}
        ]
    )
    
    current = planner.execute_step()
    print(f"Current task: {current.name if current else None}")
    
    reflection = planner.reflect(
        current.id,
        action_result="Found USGS link",
        observed_data={"link_found": "https://nas.er.usgs.gov"}
    )
    print(f"Reflection: completed={reflection.is_completed}, blocked={reflection.is_blocked}")
    
    context = planner.get_context_for_prompt()
    print(f"\nContext for prompt (first 500 chars):\n{context[:500]}...")
    
    # 6. 测试 Web Agent 控制器
    print("\n--- Testing Web Agent Controller ---")
    agent = create_web_agent(
        task_name="Find Clownfish Data",
        task_description="Find clownfish distribution data on USGS NAS database",
        initial_url="https://google.com",
        initial_subtasks=[
            {"name": "Search", "description": "Search for USGS NAS"},
            {"name": "Navigate", "description": "Go to search results"},
            {"name": "Query", "description": "Search for clownfish"},
            {"name": "Extract", "description": "Get distribution data"}
        ],
        max_steps=10
    )
    
    # 模拟执行步骤
    step_result = agent.step(
        viewport_update={
            "url": "https://google.com",
            "title": "Google",
            "accessibility_tree": "Search box, buttons..."
        },
        available_actions=[
            {"name": "click", "description": "Click an element", "parameters": {"selector": "string"}},
            {"name": "type", "description": "Type text", "parameters": {"selector": "string", "text": "string"}},
            {"name": "search", "description": "Search the web", "parameters": {"query": "string"}}
        ]
    )
    
    print(f"Step status: {step_result['status']}")
    print(f"Step number: {step_result['step']}")
    print(f"Current task: {step_result['current_task']['name'] if step_result['current_task'] else None}")
    
    # 模拟 Agent 响应
    mock_response = """
<think>
I need to search for USGS NAS database to find clownfish information.
</think>

<tool_call>
{"name": "search", "arguments": {"query": "USGS NAS clownfish invasive species"}}
</tool_call>

<memory_delta>
{"entities": [{"type": "action_result", "name": "Search Query", "properties": {"query": "USGS NAS clownfish"}}]}
</memory_delta>

<reflection>
{"task_status": "in_progress", "observations": {"query_submitted": true}, "needs_replan": false}
</reflection>
"""
    
    process_result = agent.process_response(
        agent_output=mock_response,
        action_result="Search completed, 10 results found"
    )
    
    print(f"\nProcess result: {process_result['status']}")
    print(f"Reflection: {process_result['reflection']}")
    print(f"Is complete: {process_result['is_complete']}")
    
    print(f"\nAgent status: {agent.get_status()}")
    
    # 7. 测试消息处理（原有功能）
    print("\n--- Testing Message Processor (Original Functionality) ---")
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Search for something"},
        {"role": "assistant", "content": "<think>First thought</think>\n<tool_call>{}</tool_call>"},
        {"role": "assistant", "content": "<think>Second thought</think>\n<tool_call>{}</tool_call>"}
    ]
    
    processed = process_messages_remove_percent_think(test_messages, 0.5)
    think_count = sum(
        len(re.findall(r'<think>.*?</think>', m.get('content', ''), re.DOTALL))
        for m in processed if m.get('role') == 'assistant'
    )
    print(f"After removing 50% think blocks: {think_count} remaining")
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)