"""
Kissing Number 验证程序

Kissing Number 问题的约束：所有向量两两之间的夹角必须 >= 60度
即 cos(θ) <= 0.5，等价于 4 * (v1·v2)^2 <= |v1|^2 * |v2|^2 或者 v1·v2 <= 0

用法：
    python verify_kissing.py <input_file.txt>
    
支持的输入文件格式：
    1. Python list 格式: [[1, 1, 0], [1, -1, 0], ...]
    2. numpy 数组格式:
       [[ 1.,  1.,  0.],
        [ 1., -1.,  0.],
        ...]
"""

import sys
import re
import numpy as np
from itertools import combinations


def check_angle(v1, v2):
    """
    检查两个向量的夹角是否 >= 60度
    
    Returns:
        True: 合法（夹角 >= 60度）
        False: 非法（夹角 < 60度）
    """
    inner_prod = np.dot(v1, v2)
    
    # 夹角 >= 90度，一定合法
    if inner_prod <= 0:
        return True
    
    # 检查 4 * (v1·v2)^2 <= |v1|^2 * |v2|^2
    sq_norm1 = np.dot(v1, v1)
    sq_norm2 = np.dot(v2, v2)
    
    if 4 * (inner_prod ** 2) <= sq_norm1 * sq_norm2:
        return True
    
    return False


def get_angle_degrees(v1, v2):
    """计算两个向量之间的夹角（度数）"""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # 数值稳定性处理
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def verify_kissing_number(vectors):
    """
    验证一组向量是否满足 Kissing Number 约束
    
    Args:
        vectors: list of vectors，每个向量是一个 list 或 numpy array
        
    Returns:
        is_valid: bool，是否合法
        violations: list of tuples，所有违规的向量对 (i, j, angle)
    """
    n = len(vectors)
    vectors = [np.array(v, dtype=np.float64) for v in vectors]
    
    violations = []
    
    # 检查所有向量对
    for i, j in combinations(range(n), 2):
        v1, v2 = vectors[i], vectors[j]
        
        if not check_angle(v1, v2):
            angle = get_angle_degrees(v1, v2)
            violations.append((i, j, angle))
    
    is_valid = len(violations) == 0
    return is_valid, violations


def print_result(vectors, is_valid, violations):
    """打印验证结果"""
    n = len(vectors)
    d = len(vectors[0]) if n > 0 else 0
    
    print("=" * 60)
    print("Kissing Number 验证结果")
    print("=" * 60)
    print(f"向量数量: {n}")
    print(f"向量维度: {d}")
    print(f"需要检查的向量对数: {n * (n - 1) // 2}")
    print("-" * 60)
    
    if is_valid:
        print(f"✓ 验证通过！这 {n} 个向量构成合法的 Kissing Number 配置。")
    else:
        print(f"✗ 验证失败！发现 {len(violations)} 对向量违反约束（夹角 < 60度）：")
        print()
        for i, j, angle in violations[:10]:  # 最多显示10个
            print(f"  向量 {i} 和 向量 {j}: 夹角 = {angle:.4f}° < 60°")
        if len(violations) > 10:
            print(f"  ... 还有 {len(violations) - 10} 对违规")
    
    print("=" * 60)


def load_vectors_from_file(filepath):
    """
    从文件加载向量，支持多种格式：
    1. Python list 格式: [[1, 1, 0], [1, -1, 0], ...]
    2. numpy 数组格式 (每行一个向量，带或不带逗号)
    """
    with open(filepath, 'r') as f:
        content = f.read().strip()
    
    # 预处理：将 numpy 格式转换为可解析的格式
    # 1. 移除行末的逗号后的换行，替换为 "],["
    # 2. 处理 "1." 这种浮点数格式
    
    # 检测是否是 numpy 多行格式（每行一个 [...] 向量）
    lines = content.split('\n')
    if len(lines) > 1 and lines[0].strip().startswith('[') and lines[1].strip().startswith('['):
        # numpy 多行格式
        vectors = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 移除外层的 [ 或 ]（如果是首行或末行的额外括号）
            if line.startswith('[['):
                line = line[1:]
            if line.endswith(']]'):
                line = line[:-1]
            # 移除末尾的逗号
            line = line.rstrip(',')
            # 移除方括号
            line = line.strip('[]')
            if not line:
                continue
            # 分割并解析数字
            parts = re.split(r'[,\s]+', line)
            parts = [p.strip() for p in parts if p.strip()]
            if parts:
                vector = [float(p) for p in parts]
                vectors.append(vector)
        return vectors
    else:
        # 尝试标准 Python 格式
        # 将 numpy 风格的数字 (如 1.) 转换为标准格式
        content = re.sub(r'(\d+)\.\s*([,\]\s])', r'\1.0\2', content)
        content = re.sub(r'(\d+)\.([,\]\s])', r'\1.0\2', content)
        
        # 使用 eval 解析（比 ast.literal_eval 更宽松）
        try:
            vectors = eval(content)
        except:
            # 如果失败，尝试添加逗号
            content = re.sub(r'\]\s*\[', '], [', content)
            vectors = eval(content)
        return vectors


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_kissing.py <input_file.txt>")
        print()
        print("支持的输入文件格式:")
        print("  1. Python list: [[1, 1, 0, 0], [1, -1, 0, 0], ...]")
        print("  2. numpy 数组格式:")
        print("     [[ 1.,  1.,  0.,  0.],")
        print("      [ 1., -1.,  0.,  0.],")
        print("      ...]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    try:
        vectors = load_vectors_from_file(filepath)
    except FileNotFoundError:
        print(f"错误: 文件不存在 - {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 文件格式不正确 - {e}")
        sys.exit(1)
    
    # 过滤掉全零向量
    vectors = [v for v in vectors if any(x != 0 for x in v)]
    
    if len(vectors) == 0:
        print("错误: 没有有效的向量（非零向量）")
        sys.exit(1)
    
    is_valid, violations = verify_kissing_number(vectors)
    print_result(vectors, is_valid, violations)
    
    # 返回状态码
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()

