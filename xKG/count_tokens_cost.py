# 1. 将您提供的价格信息存储在一个字典中
# 键(key)是模型名称，值(value)是另一个包含价格信息的字典
MODEL_PRICE_TABLE = {
    # 模型名称: {"in": 输入价格/百万Token, "out": 输出价格/百万Token, "currency": "货币符号"}
    "DMXAPI-DeepSeek-R1": {"in": 0.6320, "out": 2.5280, "currency": "$"},
    "gpt-4.1-mini-2025-04-14": {"in": 0.4000, "out": 1.6000, "currency": "$"},
    "gpt-4o-mini": {"in": 0.1500, "out": 0.6000, "currency": "$"},
    "o3-mini-2025-01-31": {"in": 1.1000, "out": 4.4000, "currency": "$"},
    "o4-mini-2025-04-16": {"in": 1.1000, "out": 4.4000, "currency": "$"},
    "claude-sonnet-4-20250514": {"in": 3.0000, "out": 15.0000, "currency": "$"},
    # 注意：对于没有价格的模型，我们可以不加进来，或者给一个特殊值
    # "claude-3-5-sonnet-20241022": None, 
    # "claude-3-7-sonnet-20250219": None,
    "gemini-2.5-pro-preview-06-05": {"in": 1.2500, "out": 10.0000, "currency": "$"},
    "DMXAPI-HuoShan-DeepSeek-V3": {"in": 0.3160, "out": 1.2640, "currency": "$"},
    "deepseek-v3.1-nothinking": {"in": 0.4000, "out": 1.6000, "currency": "$"},
    "text-embedding-3-small": {"in": 0.0200, "out": 0.0200, "currency": "$"},
}

def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int
) -> str:
    """
    根据给定的模型名称、输入和输出Token数量，计算并返回API调用的成本。

    Args:
        model_name (str): 要计算成本的模型名称。
        input_tokens (int): 输入的Token数量。
        output_tokens (int): 输出的Token数量。

    Returns:
        str: 格式化后的成本字符串 (例如, "$0.1234")，
             如果模型未在价格表中找到，则返回错误信息。
    """
    # 2. 检查模型是否在我们的价格表中
    if model_name not in MODEL_PRICE_TABLE:
        return f"Error: Price for model '{model_name}' not found."

    # 获取该模型的价格信息
    prices = MODEL_PRICE_TABLE[model_name]
    
    # 有些模型可能没有价格信息（例如您列表中的 claude-3-5...）
    if not prices:
        return f"Info: Price for model '{model_name}' is not available."

    # 3. 计算成本
    # 成本 = (Token数量 / 1,000,000) * 每百万Token的价格
    input_cost = (input_tokens / 1_000_000) * prices["in"]
    output_cost = (output_tokens / 1_000_000) * prices["out"]
    total_cost = input_cost + output_cost

    # 4. 格式化输出，保留4位小数
    currency_symbol = prices.get("currency", "$") # 默认使用 '$'
    return f"{currency_symbol}{total_cost:.4f}"

# --- 示例用法 ---

# 示例 1: 计算 gpt-4o-mini 的成本
model = "o4-mini-2025-04-16"
in_tokens = 255767.57  # 15万输入Token
out_tokens = 102949.57   # 5万输出Token
cost_str = calculate_cost(model, in_tokens, out_tokens)
print(f"Cost for '{model}' with {in_tokens} input tokens and {out_tokens} output tokens: {cost_str}")
# 预期计算: (150000/1M * 0.15) + (50000/1M * 0.60) = 0.0225 + 0.03 = $0.0525


