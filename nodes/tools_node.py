import random
import string


class RandomStringNode:
    """
    随机字符串生成节点
    可以按照规则随机生成字符串
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "指定长度": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "display": "number"
                }),
                "最小长度": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "display": "number"
                }),
                "最大长度": ("INT", {
                    "default": 256,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "display": "number"
                }),
                "生成指定内容": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "生成数字": ("BOOLEAN", {"default": True}),
                "生成大写字母": ("BOOLEAN", {"default": True}),
                "生成小写字母": ("BOOLEAN", {"default": True}),
                "首字母为英文": ("BOOLEAN", {"default": True}),
                "末尾不生成指定内容": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("随机字符串",)
    FUNCTION = "generate_random_string"
    CATEGORY = "CC-API/Tools"

    def generate_random_string(self, 指定长度, 最小长度, 最大长度, 生成指定内容, 生成数字, 生成大写字母, 生成小写字母, 首字母为英文, 末尾不生成指定内容, seed=0):
        try:
            # 如果seed为0，则使用当前时间作为种子，否则使用提供的seed
            if seed == 0:
                random.seed()
            else:
                random.seed(seed)
            
            # 处理生成指定内容，去除空格并分割成字符列表
            specified_chars = [char for char in 生成指定内容.replace(" ", "") if char]
            
            # 确定字符串长度
            if 指定长度 > 0:
                length = 指定长度
            else:
                # 确保最小长度不大于最大长度
                min_len = min(最小长度, 最大长度)
                max_len = max(最小长度, 最大长度)
                length = random.randint(min_len, max_len)
            
            # 构建字符集
            char_pool = []
            if 生成数字:
                char_pool.extend(string.digits)
            if 生成大写字母:
                char_pool.extend(string.ascii_uppercase)
            if 生成小写字母:
                char_pool.extend(string.ascii_lowercase)
            
            # 如果没有选择任何字符类型，则默认使用小写字母
            if not char_pool:
                char_pool = list(string.ascii_lowercase)
            
            # 生成随机字符串
            if specified_chars and not 末尾不生成指定内容:
                # 确保指定内容存在于结果中，但位置随机分布
                result_chars = []
                
                # 初始化结果数组
                for _ in range(length):
                    result_chars.append(None)
                
                # 随机选择位置放置指定内容
                available_positions = list(range(length))
                random.shuffle(available_positions)
                
                # 放置指定内容
                for char in specified_chars:
                    if available_positions:
                        pos = available_positions.pop()
                        result_chars[pos] = char
                
                # 填充剩余位置
                for i in range(length):
                    if result_chars[i] is None:
                        result_chars[i] = random.choice(char_pool)
                
                # 如果需要首字母为英文且第一个字符不是英文，则调整首字母
                if 首字母为英文 and result_chars and not result_chars[0].isalpha():
                    result_chars[0] = random.choice(string.ascii_letters)
                    
                result = ''.join(result_chars)
            elif specified_chars and 末尾不生成指定内容:
                # 生成字符串，确保末尾不包含指定内容，但其他位置可以包含
                result_chars = []
                
                # 如果需要首字母为英文，先添加一个英文字符
                if 首字母为英文:
                    first_char = random.choice(string.ascii_letters)
                    result_chars.append(first_char)
                
                # 填充到足够长度
                for i in range(length):
                    # 对于最后一个字符，确保它不在指定字符中
                    if i == length - 1:  # 最后一个字符
                        # 获取不在指定字符中的可用字符
                        available_chars = [c for c in char_pool if c not in specified_chars]
                        # 如果所有字符都在指定字符中，则从所有字符中选择
                        if not available_chars:
                            available_chars = char_pool
                        result_chars.append(random.choice(available_chars))
                    else:
                        # 非最后一个字符可以是任何字符（包括指定内容）
                        result_chars.append(random.choice(char_pool))
                
                result = ''.join(result_chars)
                
                # 确保至少包含一个指定内容（随机位置，但不能是末尾）
                if length > 1:  # 只有当长度大于1时才可能在非末尾位置放置指定内容
                    # 随机选择一个非末尾位置来放置指定内容
                    non_last_positions = list(range(length - 1))
                    random.shuffle(non_last_positions)
                    
                    # 从指定内容中随机选择一个字符放置在随机位置
                    if non_last_positions and specified_chars:
                        pos = non_last_positions[0]
                        char = random.choice(specified_chars)
                        result_chars[pos] = char
                        result = ''.join(result_chars)
            else:
                # 生成普通的随机字符串
                result_chars = []
                
                # 如果需要首字母为英文，先添加一个英文字符
                if 首字母为英文:
                    first_char = random.choice(string.ascii_letters)
                    result_chars.append(first_char)
                    # 填充剩余长度
                    for _ in range(length - 1):
                        result_chars.append(random.choice(char_pool))
                else:
                    # 填充到指定长度
                    for _ in range(length):
                        result_chars.append(random.choice(char_pool))
                
                result = ''.join(result_chars)
            
            # 确保长度正确
            if len(result) > length:
                result = result[:length]
            elif len(result) < length:
                # 补充到指定长度
                for _ in range(length - len(result)):
                    result += random.choice(char_pool)
            
            return (result,)
        except Exception as e:
            raise Exception(f"生成随机字符串时出错: {str(e)}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "RandomStringNode": RandomStringNode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomStringNode": "随机字符串"
}