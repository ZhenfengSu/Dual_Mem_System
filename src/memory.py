# memory.py

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class MemoryProcessor:
    """
    消息预处理器，负责在消息传入 Agent 之前进行处理。
    当前功能：移除 assistant 消息中的 <think>...</think> 思考过程，
    但保留 system 和 user 消息中的内容（包括格式说明）。
    """
    
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
    
    def remove_think_blocks(self, content: str) -> str:
        """
        移除文本中所有的 <think>...</think> 块。
        
        Args:
            content: 原始文本内容
            
        Returns:
            移除思考块后的文本
        """
        if not content:
            return content
        
        # 移除所有 think 块
        cleaned = self.think_pattern.sub('', content)
        
        # 清理多余的空白行（连续多个换行变成最多两个）
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # 清理开头和结尾的空白
        cleaned = cleaned.strip()
        
        return cleaned
    
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
                processed['content'] = self.remove_think_blocks(processed['content'])
        
        return processed
    
    def process_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理消息列表。只移除 assistant 消息中的思考过程，
        保留 system 和 user 消息完整不变。
        
        Args:
            messages: 消息列表
            
        Returns:
            处理后的消息列表
        """
        processed_messages = []
        
        for msg in messages:
            processed_msg = self.process_message(msg)
            
            # 保留所有消息，即使 assistant 消息处理后为空
            # 因为空的 assistant 消息可能表示某种状态
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
            'total_original_chars': original_chars,
            'total_processed_chars': processed_chars,
            'saved_chars': original_chars - processed_chars,
            'reduction_ratio': round((original_chars - processed_chars) / original_chars, 3) if original_chars > 0 else 0,
            'by_role': role_stats
        }


def create_memory_processor() -> MemoryProcessor:
    """
    工厂函数，创建 MemoryProcessor 实例。
    """
    return MemoryProcessor()


def process_messages_before_agent(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    便捷函数：处理消息列表，只移除 assistant 消息中的思考过程。
    保留 system 和 user 消息完整不变。
    
    Args:
        messages: 原始消息列表
        
    Returns:
        处理后的消息列表
    """
    processor = MemoryProcessor()
    return processor.process_messages(messages)


# 测试代码
if __name__ == "__main__":
    # 模拟你提供的真实消息格式
    test_messages = [
        {
            "role": "system", 
            "content": """You are a Web Information Seeking Master. Your task is to thoroughly seek the internet for information and provide accurate answers to questions.

Example response:
<think> thinking process here </think>
<tool_call>
{"name": "tool name here", "arguments": {"parameter name here": parameter value here}}
</tool_call>
<tool_response>
tool_response here
</tool_response>
<think> thinking process here </think>
<answer> answer here </answer>

Current date: 2025-12-11"""
        },
        {
            "role": "user", 
            "content": """A conversation between User and Assistant...

User: I'm researching species that became invasive after people who kept them as pets released them. There's a certain species of fish..."""
        },
        {
            "role": "assistant",
            "content": """<think>
The user is asking about the clownfish (Amphiprion ocellaris), which was popularized by Finding Nemo.
I need to search for USGS records of this species as a nonnative species before 2020.
Let me search for this information.
</think>

<tool_call>
{"name": "search", "arguments": {"query": ["USGS clownfish nonnative species locations", "Amphiprion ocellaris invasive USA USGS"]}}
</tool_call>"""
        },
        {
            "role": "user",
            "content": """<tool_response>
A Google search found 10 results:
1. USGS NAS Database - Amphiprion ocellaris sightings...
</tool_response>"""
        },
        {
            "role": "assistant",
            "content": """<think>
I found some results. Let me visit the USGS page to get the specific zip codes.
I need to find records before 2020.
</think>

<tool_call>
{"name": "visit", "arguments": {"url": ["https://nas.er.usgs.gov/queries/SpeciesList.aspx?ID=2334"], "goal": "Find zip codes where clownfish was found as nonnative before 2020"}}
</tool_call>"""
        }
    ]
    
    processor = MemoryProcessor()
    
    print("=" * 70)
    print("测试：只移除 assistant 消息中的 <think>，保留 system/user 消息")
    print("=" * 70)
    
    processed = processor.process_messages(test_messages)
    
    for i, (orig, proc) in enumerate(zip(test_messages, processed)):
        role = orig['role']
        orig_content = orig.get('content', '')
        proc_content = proc.get('content', '')
        
        print(f"\n[Message {i}] Role: {role}")
        print("-" * 50)
        
        if orig_content == proc_content:
            print(f"✓ 内容保持不变 (长度: {len(orig_content)} chars)")
            # 只显示前 200 字符
            preview = orig_content[:200] + "..." if len(orig_content) > 200 else orig_content
            print(f"预览: {preview}")
        else:
            print(f"✗ 内容已处理")
            print(f"  原始长度: {len(orig_content)} chars")
            print(f"  处理后长度: {len(proc_content)} chars")
            print(f"  节省: {len(orig_content) - len(proc_content)} chars")
            print(f"\n处理后内容:\n{proc_content}")
    
    # 统计信息
    stats = processor.get_stats(test_messages, processed)
    print("\n" + "=" * 70)
    print("统计信息:")
    print("=" * 70)
    print(f"总节省字符: {stats['saved_chars']} ({stats['reduction_ratio']*100:.1f}%)")
    print("\n按角色统计:")
    for role, data in stats['by_role'].items():
        saved = data['original'] - data['processed']
        print(f"  {role}: {data['original']} -> {data['processed']} (节省 {saved})")