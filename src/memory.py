# memory.py

import re
from typing import List, Dict, Any, Optional, Set, Literal
from dataclasses import dataclass, field
from copy import deepcopy
from enum import Enum


class ThinkRemovalMode(Enum):
    """思考块移除模式"""
    REMOVE_ALL = "remove_all"           # 移除所有 think 块
    KEEP_LAST = "keep_last"             # 保留最后一个 think 块
    KEEP_ALL = "keep_all"               # 保留所有（不处理）


@dataclass
class MemoryProcessor:
    """
    消息预处理器，负责在消息传入 Agent 之前进行处理。
    
    支持两种模式：
    1. REMOVE_ALL: 移除 assistant 消息中所有的 <think>...</think> 思考过程
    2. KEEP_LAST: 保留 assistant 消息中最后一个 <think>...</think>，移除之前的
    
    system 和 user 消息始终保持不变。
    """
    
    # 处理模式
    mode: ThinkRemovalMode = ThinkRemovalMode.REMOVE_ALL
    
    # 用于匹配 think 标签的正则表达式
    think_pattern: re.Pattern = field(
        default_factory=lambda: re.compile(
            r'<think>.*?</think>',
            re.DOTALL  # 使 . 匹配包括换行符在内的所有字符
        )
    )
    
    # 需要处理的角色（只处理 assistant 的消息）
    roles_to_process: Set[str] = field(
        default_factory=lambda: {'assistant'}
    )
    
    def find_all_think_blocks(self, content: str) -> List[re.Match]:
        """
        找到文本中所有的 <think>...</think> 块。
        
        Args:
            content: 原始文本内容
            
        Returns:
            所有匹配的 Match 对象列表
        """
        if not content:
            return []
        return list(self.think_pattern.finditer(content))
    
    def remove_all_think_blocks(self, content: str) -> str:
        """
        移除文本中所有的 <think>...</think> 块。
        
        Args:
            content: 原始文本内容
            
        Returns:
            移除所有思考块后的文本
        """
        if not content:
            return content
        
        # 移除所有 think 块
        cleaned = self.think_pattern.sub('', content)
        
        # 清理格式
        return self._clean_whitespace(cleaned)
    
    def remove_all_but_last_think_block(self, content: str) -> str:
        """
        移除除最后一个之外的所有 <think>...</think> 块。
        
        Args:
            content: 原始文本内容
            
        Returns:
            只保留最后一个思考块的文本
        """
        if not content:
            return content
        
        # 找到所有 think 块
        matches = self.find_all_think_blocks(content)
        
        # 如果没有或只有一个 think 块，不需要处理
        if len(matches) <= 1:
            return content
        
        # 需要移除的是除了最后一个之外的所有块
        # 从后往前处理，避免索引变化问题
        blocks_to_remove = matches[:-1]  # 除了最后一个
        
        result = content
        # 从后往前移除，这样索引不会错乱
        for match in reversed(blocks_to_remove):
            start, end = match.span()
            result = result[:start] + result[end:]
        
        # 清理格式
        return self._clean_whitespace(result)
    
    def _clean_whitespace(self, content: str) -> str:
        """
        清理文本中的多余空白。
        
        Args:
            content: 待清理的文本
            
        Returns:
            清理后的文本
        """
        # 清理多余的空白行（连续多个换行变成最多两个）
        cleaned = re.sub(r'\n{3,}', '\n\n', content)
        
        # 清理开头和结尾的空白
        cleaned = cleaned.strip()
        
        return cleaned
    
    def process_content(self, content: str) -> str:
        """
        根据当前模式处理内容。
        
        Args:
            content: 原始文本内容
            
        Returns:
            处理后的文本
        """
        if self.mode == ThinkRemovalMode.KEEP_ALL:
            return content
        elif self.mode == ThinkRemovalMode.REMOVE_ALL:
            return self.remove_all_think_blocks(content)
        elif self.mode == ThinkRemovalMode.KEEP_LAST:
            return self.remove_all_but_last_think_block(content)
        else:
            return content
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单条消息。只处理 assistant 角色的消息，
        保留 system 和 user 消息不变。
        
        Args:
            message: 单条消息字典，包含 role 和 content
            
        Returns:
            处理后的消息字典
        """
        processed = deepcopy(message)
        
        role = processed.get('role', '')
        
        # 只处理指定角色的消息
        if role in self.roles_to_process:
            if 'content' in processed and isinstance(processed['content'], str):
                processed['content'] = self.process_content(processed['content'])
        
        return processed
    
    def process_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理消息列表。根据模式处理 assistant 消息中的思考过程，
        保留 system 和 user 消息完整不变。
        
        Args:
            messages: 消息列表
            
        Returns:
            处理后的消息列表
        """
        processed_messages = []
        
        for msg in messages:
            processed_msg = self.process_message(msg)
            processed_messages.append(processed_msg)
        
        return processed_messages
    
    def get_stats(self, original_messages: List[Dict[str, Any]], 
                  processed_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取处理统计信息。
        
        Args:
            original_messages: 原始消息列表
            processed_messages: 处理后消息列表
            
        Returns:
            统计信息字典
        """
        original_chars = sum(len(m.get('content', '')) for m in original_messages)
        processed_chars = sum(len(m.get('content', '')) for m in processed_messages)
        
        # 按角色统计
        role_stats = {}
        for orig, proc in zip(original_messages, processed_messages):
            role = orig.get('role', 'unknown')
            if role not in role_stats:
                role_stats[role] = {'original': 0, 'processed': 0}
            role_stats[role]['original'] += len(orig.get('content', ''))
            role_stats[role]['processed'] += len(proc.get('content', ''))
        
        return {
            'mode': self.mode.value,
            'total_original_chars': original_chars,
            'total_processed_chars': processed_chars,
            'saved_chars': original_chars - processed_chars,
            'reduction_ratio': round((original_chars - processed_chars) / original_chars, 3) if original_chars > 0 else 0,
            'by_role': role_stats
        }


def create_memory_processor(mode: str = "remove_all") -> MemoryProcessor:
    """
    工厂函数，创建 MemoryProcessor 实例。
    
    Args:
        mode: 处理模式
            - "remove_all": 移除所有 think 块
            - "keep_last": 保留最后一个 think 块
            - "keep_all": 保留所有（不处理）
    
    Returns:
        MemoryProcessor 实例
    """
    mode_map = {
        "remove_all": ThinkRemovalMode.REMOVE_ALL,
        "keep_last": ThinkRemovalMode.KEEP_LAST,
        "keep_all": ThinkRemovalMode.KEEP_ALL,
    }
    
    think_mode = mode_map.get(mode, ThinkRemovalMode.REMOVE_ALL)
    return MemoryProcessor(mode=think_mode)


def process_messages_remove_all(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    便捷函数：移除 assistant 消息中所有的思考过程。
    
    Args:
        messages: 原始消息列表
        
    Returns:
        处理后的消息列表
    """
    processor = MemoryProcessor(mode=ThinkRemovalMode.REMOVE_ALL)
    return processor.process_messages(messages)


def process_messages_keep_last(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    便捷函数：保留 assistant 消息中最后一个思考过程，移除之前的。
    
    Args:
        messages: 原始消息列表
        
    Returns:
        处理后的消息列表
    """
    processor = MemoryProcessor(mode=ThinkRemovalMode.KEEP_LAST)
    return processor.process_messages(messages)


# 测试代码
if __name__ == "__main__":
    # 模拟真实消息格式，assistant 消息包含多个 think 块
    test_messages = [
        {
            "role": "system", 
            "content": """You are a Web Information Seeking Master.

Example response:
<think> thinking process here </think>
<tool_call>
{"name": "tool name here", "arguments": {...}}
</tool_call>
<think> thinking process here </think>
<answer> answer here </answer>

Current date: 2025-12-11"""
        },
        {
            "role": "user", 
            "content": "I'm researching species that became invasive..."
        },
        {
            "role": "assistant",
            "content": """<think>
First thought: The user is asking about clownfish from Finding Nemo.
Let me search for USGS records.
</think>

<tool_call>
{"name": "search", "arguments": {"query": ["USGS clownfish nonnative"]}}
</tool_call>

<think>
Second thought: I found some results, need to dig deeper.
The USGS database might have specific locations.
</think>

<tool_call>
{"name": "visit", "arguments": {"url": ["https://nas.er.usgs.gov/..."]}}
</tool_call>

<think>
Third thought: Now I have the data.
I can see the zip codes: 33139, 33140.
Let me verify this information before answering.
</think>

<answer>
Based on USGS records, the clownfish was found in: 33139, 33140
</answer>"""
        },
        {
            "role": "user",
            "content": "Can you verify that information?"
        },
        {
            "role": "assistant",
            "content": """<think>
The user wants verification. Let me double-check the USGS database.
I should search for more specific records.
</think>

<tool_call>
{"name": "search", "arguments": {"query": ["USGS NAS clownfish Florida zip code"]}}
</tool_call>"""
        }
    ]
    
    print("=" * 80)
    print("原始 assistant 消息内容预览")
    print("=" * 80)
    for i, msg in enumerate(test_messages):
        if msg['role'] == 'assistant':
            print(f"\n[Message {i}] assistant:")
            print("-" * 40)
            print(msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content'])
    
    # 测试模式 1: REMOVE_ALL
    print("\n" + "=" * 80)
    print("模式 1: REMOVE_ALL - 移除所有 think 块")
    print("=" * 80)
    
    processor_remove_all = MemoryProcessor(mode=ThinkRemovalMode.REMOVE_ALL)
    processed_remove_all = processor_remove_all.process_messages(test_messages)
    
    for i, (orig, proc) in enumerate(zip(test_messages, processed_remove_all)):
        if orig['role'] == 'assistant':
            print(f"\n[Message {i}] assistant 处理结果:")
            print("-" * 40)
            print(f"原始长度: {len(orig['content'])} -> 处理后: {len(proc['content'])}")
            print(f"\n处理后内容:\n{proc['content']}")
    
    stats1 = processor_remove_all.get_stats(test_messages, processed_remove_all)
    print(f"\n统计: 节省 {stats1['saved_chars']} 字符 ({stats1['reduction_ratio']*100:.1f}%)")
    
    # 测试模式 2: KEEP_LAST
    print("\n" + "=" * 80)
    print("模式 2: KEEP_LAST - 保留最后一个 think 块")
    print("=" * 80)
    
    processor_keep_last = MemoryProcessor(mode=ThinkRemovalMode.KEEP_LAST)
    processed_keep_last = processor_keep_last.process_messages(test_messages)
    
    for i, (orig, proc) in enumerate(zip(test_messages, processed_keep_last)):
        if orig['role'] == 'assistant':
            print(f"\n[Message {i}] assistant 处理结果:")
            print("-" * 40)
            print(f"原始长度: {len(orig['content'])} -> 处理后: {len(proc['content'])}")
            print(f"\n处理后内容:\n{proc['content']}")
    
    stats2 = processor_keep_last.get_stats(test_messages, processed_keep_last)
    print(f"\n统计: 节省 {stats2['saved_chars']} 字符 ({stats2['reduction_ratio']*100:.1f}%)")
    
    # 对比两种模式
    print("\n" + "=" * 80)
    print("两种模式对比")
    print("=" * 80)
    print(f"REMOVE_ALL: 节省 {stats1['saved_chars']} 字符 ({stats1['reduction_ratio']*100:.1f}%)")
    print(f"KEEP_LAST:  节省 {stats2['saved_chars']} 字符 ({stats2['reduction_ratio']*100:.1f}%)")
    
    # 测试便捷函数
    print("\n" + "=" * 80)
    print("便捷函数测试")
    print("=" * 80)
    
    result1 = process_messages_remove_all(test_messages)
    result2 = process_messages_keep_last(test_messages)
    
    print(f"process_messages_remove_all: 处理后总字符 {sum(len(m['content']) for m in result1)}")
    print(f"process_messages_keep_last:  处理后总字符 {sum(len(m['content']) for m in result2)}")
    
    # 测试工厂函数
    print("\n" + "=" * 80)
    print("工厂函数测试")
    print("=" * 80)
    
    p1 = create_memory_processor("remove_all")
    p2 = create_memory_processor("keep_last")
    p3 = create_memory_processor("keep_all")
    
    print(f"create_memory_processor('remove_all'): mode = {p1.mode}")
    print(f"create_memory_processor('keep_last'):  mode = {p2.mode}")
    print(f"create_memory_processor('keep_all'):   mode = {p3.mode}")