# memory.py

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from copy import deepcopy
from enum import Enum


class ThinkRemovalMode(Enum):
    """思考块移除模式"""
    REMOVE_ALL = "remove_all"           # 移除所有 think 块
    KEEP_LAST = "keep_last"             # 所有 assistant 消息中只保留最后一个 think 块
    KEEP_ALL = "keep_all"               # 保留所有（不处理）


class ToolCallRemovalMode(Enum):
    """工具调用块移除模式"""
    REMOVE_ALL = "remove_all"           # 移除所有 tool_call 块
    KEEP_LAST = "keep_last"             # 所有 assistant 消息中只保留最后一个 tool_call 块
    KEEP_ALL = "keep_all"               # 保留所有（不处理）


@dataclass
class MemoryProcessor:
    """
    消息预处理器，负责在消息传入 Agent 之前进行处理。
    
    支持对 <think> 块的处理模式：
    1. REMOVE_ALL: 移除所有 assistant 消息中所有的 <think>...</think>
    2. KEEP_LAST: 所有 assistant 消息中只保留最后一个 <think>...</think>
    3. KEEP_ALL: 保留所有（不处理）
    
    支持对 <tool_call> 块的处理模式：
    1. REMOVE_ALL: 移除所有 assistant 消息中所有的 <tool_call>...</tool_call>
    2. KEEP_LAST: 所有 assistant 消息中只保留最后一个 <tool_call>...</tool_call>
    3. KEEP_ALL: 保留所有（不处理）
    
    注意：KEEP_LAST 是跨所有 assistant 消息的，即整个对话中只保留最后一个。
    
    system 和 user 消息始终保持不变。
    """
    
    # think 处理模式
    think_mode: ThinkRemovalMode = ThinkRemovalMode.REMOVE_ALL
    
    # tool_call 处理模式
    tool_call_mode: ToolCallRemovalMode = ToolCallRemovalMode.KEEP_ALL
    
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
        last_block = all_blocks[-1]
        last_msg_idx, last_match = last_block
        blocks_to_remove = all_blocks[:-1]
        
        # 3. 按消息分组需要移除的块
        # {msg_idx: [match1, match2, ...]}
        remove_by_msg: Dict[int, List[re.Match]] = {}
        for msg_idx, match in blocks_to_remove:
            if msg_idx not in remove_by_msg:
                remove_by_msg[msg_idx] = []
            remove_by_msg[msg_idx].append(match)
        
        # 4. 处理每条消息
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
        mode: Enum
    ) -> List[Dict[str, Any]]:
        """根据模式处理指定类型的块"""
        if mode.value == "keep_all":
            return deepcopy(messages)
        elif mode.value == "remove_all":
            return self._process_messages_remove_all(messages, pattern)
        elif mode.value == "keep_last":
            return self._process_messages_keep_last(messages, pattern)
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
            self.think_mode
        )
        
        # 再处理 tool_call 块
        result = self._process_for_block_type(
            result, 
            self.tool_call_pattern, 
            self.tool_call_mode
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
            'total_original_chars': original_chars,
            'total_processed_chars': processed_chars,
            'saved_chars': original_chars - processed_chars,
            'reduction_ratio': round((original_chars - processed_chars) / original_chars, 3) if original_chars > 0 else 0,
            'think_blocks': {
                'original': original_think_count,
                'processed': processed_think_count
            },
            'tool_call_blocks': {
                'original': original_tool_call_count,
                'processed': processed_tool_call_count
            }
        }


# ============================================================
# 工厂函数
# ============================================================

def create_memory_processor(
    think_mode: str = "remove_all",
    tool_call_mode: str = "keep_all"
) -> MemoryProcessor:
    """
    工厂函数，创建 MemoryProcessor 实例。
    
    Args:
        think_mode: think 处理模式 ("remove_all", "keep_last", "keep_all")
        tool_call_mode: tool_call 处理模式 ("remove_all", "keep_last", "keep_all")
    
    Returns:
        MemoryProcessor 实例
    """
    think_mode_map = {
        "remove_all": ThinkRemovalMode.REMOVE_ALL,
        "keep_last": ThinkRemovalMode.KEEP_LAST,
        "keep_all": ThinkRemovalMode.KEEP_ALL,
    }
    
    tool_call_mode_map = {
        "remove_all": ToolCallRemovalMode.REMOVE_ALL,
        "keep_last": ToolCallRemovalMode.KEEP_LAST,
        "keep_all": ToolCallRemovalMode.KEEP_ALL,
    }
    
    return MemoryProcessor(
        think_mode=think_mode_map.get(think_mode, ThinkRemovalMode.REMOVE_ALL),
        tool_call_mode=tool_call_mode_map.get(tool_call_mode, ToolCallRemovalMode.KEEP_ALL)
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
    
    # 测试各种模式
    test_cases = [
        ("移除所有 think, 保留所有 tool_call", ThinkRemovalMode.REMOVE_ALL, ToolCallRemovalMode.KEEP_ALL),
        ("保留最后 think, 保留所有 tool_call", ThinkRemovalMode.KEEP_LAST, ToolCallRemovalMode.KEEP_ALL),
        ("保留所有 think, 移除所有 tool_call", ThinkRemovalMode.KEEP_ALL, ToolCallRemovalMode.REMOVE_ALL),
        ("保留所有 think, 保留最后 tool_call", ThinkRemovalMode.KEEP_ALL, ToolCallRemovalMode.KEEP_LAST),
        ("移除所有 think, 移除所有 tool_call", ThinkRemovalMode.REMOVE_ALL, ToolCallRemovalMode.REMOVE_ALL),
        ("保留最后 think, 保留最后 tool_call", ThinkRemovalMode.KEEP_LAST, ToolCallRemovalMode.KEEP_LAST),
    ]
    
    for desc, think_mode, tool_call_mode in test_cases:
        print("\n" + "=" * 80)
        print(f"模式: {desc}")
        print(f"       think_mode={think_mode.value}, tool_call_mode={tool_call_mode.value}")
        print("=" * 80)
        
        processor = MemoryProcessor(think_mode=think_mode, tool_call_mode=tool_call_mode)
        processed = processor.process_messages(test_messages)
        stats = processor.get_stats(test_messages, processed)
        
        print(f"\nthink 块: {stats['think_blocks']['original']} -> {stats['think_blocks']['processed']}")
        print(f"tool_call 块: {stats['tool_call_blocks']['original']} -> {stats['tool_call_blocks']['processed']}")
        print(f"节省字符: {stats['saved_chars']} ({stats['reduction_ratio']*100:.1f}%)")
        
        print("\n处理后的 assistant 消息:")
        for i, msg in enumerate(processed):
            if msg['role'] == 'assistant':
                print(f"\n[Message {i}] assistant:")
                print("-" * 40)
                print(msg['content'] if msg['content'] else "(空)")
    
    # 特别验证 keep_last 行为
    print("\n" + "=" * 80)
    print("验证 KEEP_LAST 跨消息行为")
    print("=" * 80)
    
    # keep_last think
    processor = MemoryProcessor(think_mode=ThinkRemovalMode.KEEP_LAST, tool_call_mode=ToolCallRemovalMode.KEEP_ALL)
    processed = processor.process_messages(test_messages)
    
    remaining_thinks = []
    for i, msg in enumerate(processed):
        if msg['role'] == 'assistant':
            thinks = processor.think_pattern.findall(msg['content'])
            for t in thinks:
                remaining_thinks.append((i, t[:50] + "..."))
    
    print(f"\nKEEP_LAST think 后，剩余 {len(remaining_thinks)} 个 think 块:")
    for msg_idx, content_preview in remaining_thinks:
        print(f"  Message {msg_idx}: {content_preview}")
    
    # keep_last tool_call
    processor = MemoryProcessor(think_mode=ThinkRemovalMode.KEEP_ALL, tool_call_mode=ToolCallRemovalMode.KEEP_LAST)
    processed = processor.process_messages(test_messages)
    
    remaining_tool_calls = []
    for i, msg in enumerate(processed):
        if msg['role'] == 'assistant':
            tool_calls = processor.tool_call_pattern.findall(msg['content'])
            for t in tool_calls:
                remaining_tool_calls.append((i, t[:50] + "..."))
    
    print(f"\nKEEP_LAST tool_call 后，剩余 {len(remaining_tool_calls)} 个 tool_call 块:")
    for msg_idx, content_preview in remaining_tool_calls:
        print(f"  Message {msg_idx}: {content_preview}")