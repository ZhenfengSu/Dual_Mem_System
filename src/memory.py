# memory.py

import re
import math
from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field
from copy import deepcopy
from enum import Enum


class ThinkRemovalMode(Enum):
    """思考块移除模式"""
    REMOVE_ALL = "remove_all"           # 移除所有 think 块
    KEEP_LAST = "keep_last"             # 所有 assistant 消息中只保留最后一个 think 块
    KEEP_ALL = "keep_all"               # 保留所有（不处理）
    REMOVE_PERCENT = "remove_percent"   # 按百分比移除前 N% 的 think 块


class ToolCallRemovalMode(Enum):
    """工具调用块移除模式"""
    REMOVE_ALL = "remove_all"           # 移除所有 tool_call 块
    KEEP_LAST = "keep_last"             # 所有 assistant 消息中只保留最后一个 tool_call 块
    KEEP_ALL = "keep_all"               # 保留所有（不处理）
    REMOVE_PERCENT = "remove_percent"   # 按百分比移除前 N% 的 tool_call 块


@dataclass
class MemoryProcessor:
    """
    消息预处理器，负责在消息传入 Agent 之前进行处理。
    
    支持对 <think> 块的处理模式：
    1. REMOVE_ALL: 移除所有 assistant 消息中所有的 <think>...</think>
    2. KEEP_LAST: 所有 assistant 消息中只保留最后一个 <think>...</think>
    3. KEEP_ALL: 保留所有（不处理）
    4. REMOVE_PERCENT: 按百分比移除前 N% 的 think 块（向上取整）
    
    支持对 <tool_call> 块的处理模式：
    1. REMOVE_ALL: 移除所有 assistant 消息中所有的 <tool_call>...</tool_call>
    2. KEEP_LAST: 所有 assistant 消息中只保留最后一个 <tool_call>...</tool_call>
    3. KEEP_ALL: 保留所有（不处理）
    4. REMOVE_PERCENT: 按百分比移除前 N% 的 tool_call 块（向上取整）
    
    注意：KEEP_LAST 和 REMOVE_PERCENT 是跨所有 assistant 消息的。
    
    system 和 user 消息始终保持不变。
    """
    
    # think 处理模式
    think_mode: ThinkRemovalMode = ThinkRemovalMode.REMOVE_ALL
    
    # tool_call 处理模式
    tool_call_mode: ToolCallRemovalMode = ToolCallRemovalMode.KEEP_ALL
    
    # think 移除百分比 (0.0 - 1.0)，仅在 REMOVE_PERCENT 模式下生效
    think_remove_percent: float = 1.0
    
    # tool_call 移除百分比 (0.0 - 1.0)，仅在 REMOVE_PERCENT 模式下生效
    tool_call_remove_percent: float = 1.0
    
    # 用于匹配 think 标签的正则表达式
    think_pattern: re.Pattern = field(
        default_factory=lambda: re.compile(
            r'<think>.*?</think>',
            re.DOTALL
        )
    )
    
    # 用于匹配 tool_call 标签的正则表达式
    tool_call_pattern: re.Pattern = field(
        default_factory=lambda: re.compile(
            r'<tool_call>.*?</tool_call>',
            re.DOTALL
        )
    )
    
    # 需要处理的角色（只处理 assistant 的消息）
    roles_to_process: Set[str] = field(
        default_factory=lambda: {'assistant'}
    )
    
    def __post_init__(self):
        """验证百分比参数"""
        if not 0.0 <= self.think_remove_percent <= 1.0:
            raise ValueError(f"think_remove_percent must be between 0.0 and 1.0, got {self.think_remove_percent}")
        if not 0.0 <= self.tool_call_remove_percent <= 1.0:
            raise ValueError(f"tool_call_remove_percent must be between 0.0 and 1.0, got {self.tool_call_remove_percent}")
    
    def _find_all_blocks_in_message(self, content: str, pattern: re.Pattern) -> List[re.Match]:
        """找到单条消息中所有匹配的块"""
        if not content:
            return []
        return list(pattern.finditer(content))
    
    def _remove_all_blocks(self, content: str, pattern: re.Pattern) -> str:
        """移除文本中所有匹配的块"""
        if not content:
            return content
        return pattern.sub('', content)
    
    def _clean_whitespace(self, content: str) -> str:
        """清理文本中的多余空白"""
        cleaned = re.sub(r'\n{3,}', '\n\n', content)
        cleaned = cleaned.strip()
        return cleaned
    
    def _find_all_blocks_across_messages(
        self, 
        messages: List[Dict[str, Any]], 
        pattern: re.Pattern
    ) -> List[tuple]:
        """
        跨所有 assistant 消息找到所有匹配的块。
        
        Returns:
            List of (message_index, match_object) tuples
        """
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
        self, 
        messages: List[Dict[str, Any]], 
        pattern: re.Pattern
    ) -> List[Dict[str, Any]]:
        """移除所有 assistant 消息中的所有匹配块"""
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
        self, 
        messages: List[Dict[str, Any]], 
        pattern: re.Pattern
    ) -> List[Dict[str, Any]]:
        """
        跨所有 assistant 消息只保留最后一个匹配块。
        其他所有匹配块都会被移除。
        """
        # 1. 找到所有块的位置
        all_blocks = self._find_all_blocks_across_messages(messages, pattern)
        
        if len(all_blocks) <= 1:
            # 只有 0 或 1 个块，不需要处理
            return deepcopy(messages)
        
        # 2. 最后一个块需要保留，其他的需要移除
        blocks_to_remove = all_blocks[:-1]
        
        return self._remove_specified_blocks(messages, blocks_to_remove)
    
    def _process_messages_remove_percent(
        self, 
        messages: List[Dict[str, Any]], 
        pattern: re.Pattern,
        percent: float
    ) -> List[Dict[str, Any]]:
        """
        按百分比移除前 N% 的匹配块（向上取整）。
        
        Args:
            messages: 消息列表
            pattern: 正则表达式模式
            percent: 移除百分比 (0.0 - 1.0)
            
        Returns:
            处理后的消息列表
        """
        if percent <= 0:
            return deepcopy(messages)
        
        if percent >= 1.0:
            return self._process_messages_remove_all(messages, pattern)
        
        # 1. 找到所有块的位置
        all_blocks = self._find_all_blocks_across_messages(messages, pattern)
        
        if len(all_blocks) == 0:
            return deepcopy(messages)
        
        # 2. 计算需要移除的数量（向上取整）
        num_to_remove = math.ceil(len(all_blocks) * percent)
        
        if num_to_remove == 0:
            return deepcopy(messages)
        
        if num_to_remove >= len(all_blocks):
            return self._process_messages_remove_all(messages, pattern)
        
        # 3. 移除前 N 个块
        blocks_to_remove = all_blocks[:num_to_remove]
        
        return self._remove_specified_blocks(messages, blocks_to_remove)
    
    def _remove_specified_blocks(
        self,
        messages: List[Dict[str, Any]],
        blocks_to_remove: List[tuple]
    ) -> List[Dict[str, Any]]:
        """
        移除指定的块。
        
        Args:
            messages: 消息列表
            blocks_to_remove: 要移除的块列表 [(msg_idx, match), ...]
            
        Returns:
            处理后的消息列表
        """
        if not blocks_to_remove:
            return deepcopy(messages)
        
        # 按消息分组需要移除的块
        # {msg_idx: [match1, match2, ...]}
        remove_by_msg: Dict[int, List[re.Match]] = {}
        for msg_idx, match in blocks_to_remove:
            if msg_idx not in remove_by_msg:
                remove_by_msg[msg_idx] = []
            remove_by_msg[msg_idx].append(match)
        
        # 处理每条消息
        result = []
        for msg_idx, msg in enumerate(messages):
            processed = deepcopy(msg)
            
            if msg.get('role') in self.roles_to_process:
                content = msg.get('content', '')
                if isinstance(content, str) and msg_idx in remove_by_msg:
                    # 从后往前移除，避免索引偏移
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
        self, 
        messages: List[Dict[str, Any]], 
        pattern: re.Pattern, 
        mode: Enum,
        percent: float = 1.0
    ) -> List[Dict[str, Any]]:
        """根据模式处理指定类型的块"""
        if mode.value == "keep_all":
            return deepcopy(messages)
        elif mode.value == "remove_all":
            return self._process_messages_remove_all(messages, pattern)
        elif mode.value == "keep_last":
            return self._process_messages_keep_last(messages, pattern)
        elif mode.value == "remove_percent":
            return self._process_messages_remove_percent(messages, pattern, percent)
        else:
            return deepcopy(messages)
    
    def process_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理消息列表。依次处理 think 块和 tool_call 块。
        
        Args:
            messages: 原始消息列表
            
        Returns:
            处理后的消息列表
        """
        # 先处理 think 块
        result = self._process_for_block_type(
            messages, 
            self.think_pattern, 
            self.think_mode,
            self.think_remove_percent
        )
        
        # 再处理 tool_call 块
        result = self._process_for_block_type(
            result, 
            self.tool_call_pattern, 
            self.tool_call_mode,
            self.tool_call_remove_percent
        )
        
        return result
    
    def get_stats(
        self, 
        original_messages: List[Dict[str, Any]], 
        processed_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """获取处理统计信息"""
        original_chars = sum(len(m.get('content', '')) for m in original_messages)
        processed_chars = sum(len(m.get('content', '')) for m in processed_messages)
        
        # 统计原始消息中各类块的数量
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
# 工厂函数
# ============================================================

def create_memory_processor(
    think_mode: str = "remove_all",
    tool_call_mode: str = "keep_all",
    think_remove_percent: float = 1.0,
    tool_call_remove_percent: float = 1.0
) -> MemoryProcessor:
    """
    工厂函数，创建 MemoryProcessor 实例。
    
    Args:
        think_mode: think 处理模式 ("remove_all", "keep_last", "keep_all", "remove_percent")
        tool_call_mode: tool_call 处理模式 ("remove_all", "keep_last", "keep_all", "remove_percent")
        think_remove_percent: think 移除百分比 (0.0 - 1.0)，仅在 remove_percent 模式下生效
        tool_call_remove_percent: tool_call 移除百分比 (0.0 - 1.0)，仅在 remove_percent 模式下生效
    
    Returns:
        MemoryProcessor 实例
    """
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


# ============================================================
# 便捷函数 - Think 处理
# ============================================================

def process_messages_remove_all_think(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """移除所有 assistant 消息中所有的 think 块。"""
    processor = MemoryProcessor(
        think_mode=ThinkRemovalMode.REMOVE_ALL,
        tool_call_mode=ToolCallRemovalMode.KEEP_ALL
    )
    return processor.process_messages(messages)


def process_messages_keep_last_think(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """所有 assistant 消息中只保留最后一个 think 块。"""
    processor = MemoryProcessor(
        think_mode=ThinkRemovalMode.KEEP_LAST,
        tool_call_mode=ToolCallRemovalMode.KEEP_ALL
    )
    return processor.process_messages(messages)


def process_messages_remove_percent_think(
    messages: List[Dict[str, Any]], 
    percent: float = 0.75
) -> List[Dict[str, Any]]:
    """按百分比移除前 N% 的 think 块（向上取整）。"""
    processor = MemoryProcessor(
        think_mode=ThinkRemovalMode.REMOVE_PERCENT,
        tool_call_mode=ToolCallRemovalMode.KEEP_ALL,
        think_remove_percent=percent
    )
    return processor.process_messages(messages)


# ============================================================
# 便捷函数 - Tool Call 处理
# ============================================================

def process_messages_remove_all_tool_call(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """移除所有 assistant 消息中所有的 tool_call 块。"""
    processor = MemoryProcessor(
        think_mode=ThinkRemovalMode.KEEP_ALL,
        tool_call_mode=ToolCallRemovalMode.REMOVE_ALL
    )
    return processor.process_messages(messages)


def process_messages_keep_last_tool_call(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """所有 assistant 消息中只保留最后一个 tool_call 块。"""
    processor = MemoryProcessor(
        think_mode=ThinkRemovalMode.KEEP_ALL,
        tool_call_mode=ToolCallRemovalMode.KEEP_LAST
    )
    return processor.process_messages(messages)


def process_messages_remove_percent_tool_call(
    messages: List[Dict[str, Any]], 
    percent: float = 0.75
) -> List[Dict[str, Any]]:
    """按百分比移除前 N% 的 tool_call 块（向上取整）。"""
    processor = MemoryProcessor(
        think_mode=ThinkRemovalMode.KEEP_ALL,
        tool_call_mode=ToolCallRemovalMode.REMOVE_PERCENT,
        tool_call_remove_percent=percent
    )
    return processor.process_messages(messages)


# ============================================================
# 便捷函数 - 组合处理
# ============================================================

def process_messages_remove_all(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """移除所有 assistant 消息中所有的 think 块（保留 tool_call）。兼容旧接口。"""
    return process_messages_remove_all_think(messages)


def process_messages_keep_last(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """所有 assistant 消息中只保留最后一个 think 块（保留 tool_call）。兼容旧接口。"""
    return process_messages_keep_last_think(messages)


def process_messages_remove_all_think_and_tool_call(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """移除所有 assistant 消息中所有的 think 和 tool_call 块。"""
    processor = MemoryProcessor(
        think_mode=ThinkRemovalMode.REMOVE_ALL,
        tool_call_mode=ToolCallRemovalMode.REMOVE_ALL
    )
    return processor.process_messages(messages)


def process_messages_keep_last_think_and_tool_call(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """所有 assistant 消息中只保留最后一个 think 和最后一个 tool_call 块。"""
    processor = MemoryProcessor(
        think_mode=ThinkRemovalMode.KEEP_LAST,
        tool_call_mode=ToolCallRemovalMode.KEEP_LAST
    )
    return processor.process_messages(messages)


def process_messages_remove_percent_think_and_tool_call(
    messages: List[Dict[str, Any]],
    think_percent: float = 0.75,
    tool_call_percent: float = 0.75
) -> List[Dict[str, Any]]:
    """按百分比移除前 N% 的 think 和 tool_call 块。"""
    processor = MemoryProcessor(
        think_mode=ThinkRemovalMode.REMOVE_PERCENT,
        tool_call_mode=ToolCallRemovalMode.REMOVE_PERCENT,
        think_remove_percent=think_percent,
        tool_call_remove_percent=tool_call_percent
    )
    return processor.process_messages(messages)


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    test_messages = [
        {
            "role": "system", 
            "content": """You are a Web Information Seeking Master.

Example response:
<think> thinking process here </think>
<tool_call>
{"name": "tool name here", "arguments": {...}}
</tool_call>

Current date: 2025-12-11"""
        },
        {
            "role": "user", 
            "content": "I'm researching species that became invasive..."
        },
        {
            "role": "assistant",
            "content": """<think>
First thought: The user is asking about clownfish.
</think>

<tool_call>
{"name": "search", "arguments": {"query": ["USGS clownfish nonnative"]}}
</tool_call>"""
        },
        {
            "role": "user",
            "content": "<tool_response>\nSearch results here...\n</tool_response>"
        },
        {
            "role": "assistant",
            "content": """<think>
Second thought: Found some results, need to visit.
</think>

<tool_call>
{"name": "visit", "arguments": {"url": ["https://nas.er.usgs.gov/..."]}}
</tool_call>"""
        },
        {
            "role": "user",
            "content": "<tool_response>\nVisit results here...\n</tool_response>"
        },
        {
            "role": "assistant",
            "content": """<think>
Third thought: Now I have the data.
</think>

<tool_call>
{"name": "search", "arguments": {"query": ["verify clownfish data"]}}
</tool_call>"""
        },
        {
            "role": "user",
            "content": "<tool_response>\nVerification results...\n</tool_response>"
        },
        {
            "role": "assistant",
            "content": """<think>
Fourth thought: Final analysis complete.
</think>

<tool_call>
{"name": "final_check", "arguments": {"data": "..."}}
</tool_call>

<answer>
Based on USGS records, the clownfish was found in: 33139, 33140
</answer>"""
        }
    ]
    
    print("=" * 80)
    print("原始 assistant 消息内容")
    print("=" * 80)
    for i, msg in enumerate(test_messages):
        if msg['role'] == 'assistant':
            print(f"\n[Message {i}] assistant:")
            print("-" * 40)
            print(msg['content'])
    
    # 统计原始块数量
    processor = MemoryProcessor()
    original_think = sum(
        len(processor.think_pattern.findall(m.get('content', '')))
        for m in test_messages if m.get('role') == 'assistant'
    )
    original_tool_call = sum(
        len(processor.tool_call_pattern.findall(m.get('content', '')))
        for m in test_messages if m.get('role') == 'assistant'
    )
    print(f"\n原始统计: {original_think} 个 think 块, {original_tool_call} 个 tool_call 块")
    
    # ========================================
    # 测试 REMOVE_PERCENT 模式
    # ========================================
    print("\n" + "=" * 80)
    print("测试 REMOVE_PERCENT 模式 (百分比丢弃)")
    print("=" * 80)
    
    percent_test_cases = [
        (0.0, "0%"),
        (0.25, "25%"),
        (0.5, "50%"),
        (0.75, "75%"),
        (1.0, "100%"),
    ]
    
    for percent, label in percent_test_cases:
        print(f"\n--- tool_call 移除 {label} (向上取整) ---")
        processor = MemoryProcessor(
            think_mode=ThinkRemovalMode.KEEP_ALL,
            tool_call_mode=ToolCallRemovalMode.REMOVE_PERCENT,
            tool_call_remove_percent=percent
        )
        processed = processor.process_messages(test_messages)
        stats = processor.get_stats(test_messages, processed)
        
        num_to_remove = math.ceil(original_tool_call * percent)
        print(f"总共 {original_tool_call} 个 tool_call, 移除 {percent*100:.0f}% = ceil({original_tool_call} * {percent}) = {num_to_remove} 个")
        print(f"结果: {stats['tool_call_blocks']['original']} -> {stats['tool_call_blocks']['processed']} (移除了 {stats['tool_call_blocks']['removed']} 个)")
        
        # 显示保留的 tool_call 在哪些消息中
        remaining = []
        for i, msg in enumerate(processed):
            if msg['role'] == 'assistant':
                tool_calls = processor.tool_call_pattern.findall(msg['content'])
                if tool_calls:
                    remaining.append(f"Message {i}: {len(tool_calls)} 个")
        if remaining:
            print(f"保留位置: {', '.join(remaining)}")
    
    # ========================================
    # 测试 think 的 REMOVE_PERCENT 模式
    # ========================================
    print("\n" + "=" * 80)
    print("测试 think 的 REMOVE_PERCENT 模式")
    print("=" * 80)
    
    for percent, label in [(0.5, "50%"), (0.75, "75%")]:
        print(f"\n--- think 移除 {label} ---")
        processor = MemoryProcessor(
            think_mode=ThinkRemovalMode.REMOVE_PERCENT,
            tool_call_mode=ToolCallRemovalMode.KEEP_ALL,
            think_remove_percent=percent
        )
        processed = processor.process_messages(test_messages)
        stats = processor.get_stats(test_messages, processed)
        
        print(f"think 块: {stats['think_blocks']['original']} -> {stats['think_blocks']['processed']}")
    
    # ========================================
    # 测试组合模式
    # ========================================
    print("\n" + "=" * 80)
    print("测试组合模式: think 移除 50%, tool_call 移除 75%")
    print("=" * 80)
    
    processor = MemoryProcessor(
        think_mode=ThinkRemovalMode.REMOVE_PERCENT,
        tool_call_mode=ToolCallRemovalMode.REMOVE_PERCENT,
        think_remove_percent=0.5,
        tool_call_remove_percent=0.75
    )
    processed = processor.process_messages(test_messages)
    stats = processor.get_stats(test_messages, processed)
    
    print(f"think 块: {stats['think_blocks']['original']} -> {stats['think_blocks']['processed']}")
    print(f"tool_call 块: {stats['tool_call_blocks']['original']} -> {stats['tool_call_blocks']['processed']}")
    print(f"节省字符: {stats['saved_chars']} ({stats['reduction_ratio']*100:.1f}%)")
    
    print("\n处理后的 assistant 消息:")
    for i, msg in enumerate(processed):
        if msg['role'] == 'assistant':
            print(f"\n[Message {i}] assistant:")
            print("-" * 40)
            print(msg['content'] if msg['content'] else "(空)")
    
    # ========================================
    # 测试便捷函数
    # ========================================
    print("\n" + "=" * 80)
    print("测试便捷函数")
    print("=" * 80)
    
    # 测试 process_messages_remove_percent_tool_call
    result = process_messages_remove_percent_tool_call(test_messages, 0.75)
    remaining_count = sum(
        len(re.findall(r'<tool_call>.*?</tool_call>', m.get('content', ''), re.DOTALL))
        for m in result if m.get('role') == 'assistant'
    )
    print(f"process_messages_remove_percent_tool_call(75%): {original_tool_call} -> {remaining_count}")
    
    # 测试 process_messages_remove_percent_think
    result = process_messages_remove_percent_think(test_messages, 0.5)
    remaining_count = sum(
        len(re.findall(r'<think>.*?</think>', m.get('content', ''), re.DOTALL))
        for m in result if m.get('role') == 'assistant'
    )
    print(f"process_messages_remove_percent_think(50%): {original_think} -> {remaining_count}")
    
    # 测试工厂函数
    print("\n--- 工厂函数测试 ---")
    p = create_memory_processor(
        think_mode="remove_percent",
        tool_call_mode="remove_percent",
        think_remove_percent=0.5,
        tool_call_remove_percent=0.75
    )
    print(f"think_mode: {p.think_mode.value}, think_remove_percent: {p.think_remove_percent}")
    print(f"tool_call_mode: {p.tool_call_mode.value}, tool_call_remove_percent: {p.tool_call_remove_percent}")
    
    # ========================================
    # 边界情况测试
    # ========================================
    print("\n" + "=" * 80)
    print("边界情况测试")
    print("=" * 80)
    
    # 测试只有 1 个 tool_call 时的各种百分比
    single_tool_call_messages = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "<tool_call>only one</tool_call>"}
    ]
    
    for percent in [0.0, 0.25, 0.5, 0.75, 1.0]:
        processor = MemoryProcessor(
            think_mode=ThinkRemovalMode.KEEP_ALL,
            tool_call_mode=ToolCallRemovalMode.REMOVE_PERCENT,
            tool_call_remove_percent=percent
        )
        result = processor.process_messages(single_tool_call_messages)
        remaining = len(processor.tool_call_pattern.findall(result[1]['content']))
        num_to_remove = math.ceil(1 * percent)
        print(f"1 个 tool_call, 移除 {percent*100:.0f}%: ceil(1 * {percent}) = {num_to_remove}, 剩余 {remaining}")