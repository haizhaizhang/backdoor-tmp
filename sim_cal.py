import torch
import time
import pandas as pd
import torch.nn.functional as F

def compute_cosine_similarity_matrix(data: torch.Tensor,save_path: str):
    """
    features: torch.Tensor, shape (N, D)
    return: torch.Tensor, shape (N, N), where each [i, j] is cosine similarity between features[i] and features[j]
    """
    features = data["features"].to('cuda')  # shape: (N, D)
    filenames = data["filenames"]  # list of N strings

    # L2 归一化
    features = F.normalize(features, dim=1)

    # 计算余弦相似度矩阵
    sim_matrix = features @ features.T  # shape: (N, N)

    # 转换为 pandas.DataFrame
    df = pd.DataFrame(sim_matrix.cpu().numpy(), index=filenames, columns=filenames)

    # 保存为 CSV
    df.to_csv(save_path)
    print(f"已保存为{save_path}")

import torch
import json

def analyze_target_similarity(
    features_path: str,
    target_txt_path: str,
    output_path: str = "similarity_results.json",
    similarity_threshold: float = 0.8,
    save_format: str = "txt"  # 可选 "json" 或 "txt"
):
    """
    计算目标图像与非目标图像的余弦相似度，统计超过阈值的数量，并保存排序后的统计结果。
    """

    # === 加载特征数据 ===
    data = torch.load(features_path)
    features = data['features'].to('cuda')  # shape: (N, D)
    filenames = data['filenames']  # list of N strings
    features = F.normalize(features, dim=1)

    
    # === 加载目标图像名称 ===
    with open(target_txt_path, 'r') as f:
        target_names = set(line.strip() for line in f if line.strip())
    target_names = ['data/VOCdevkit/trainval/' + t for t in target_names]
    # === 构建索引映射 ===
    filename2idx = {name: i for i, name in enumerate(filenames)}
    target_indices = [filename2idx[name] for name in target_names if name in filename2idx]
    non_target_indices = [i for i in range(len(filenames)) if filenames[i] not in target_names]

    if not target_indices:
        raise ValueError("目标图像未匹配到任何有效文件名，请检查 target_txt 是否正确。")

    # === 拆分特征 ===
    non_target_features = features[non_target_indices]

    # === 相似度计算 ===
    start_time = time.time()
    
    results = {}
    for t_idx in target_indices:
        t_name = filenames[t_idx]
        t_feat = features[t_idx].unsqueeze(0)
        sim = torch.matmul(t_feat, non_target_features.T).squeeze(0)
        count = (sim > similarity_threshold).sum().item()
        results[t_name] = count
    duration = time.time() - start_time
    print(f"相似度计算完成，耗时 {duration:.2f} 秒")
    # # === 排序 ===
    # sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    # # === 保存结果 ===
    # if save_format == "json":
    #     with open(output_path, 'w') as f:
    #         json.dump(sorted_results, f, indent=2)
    # elif save_format == "txt":
    #     with open(output_path, 'w') as f:
    #         for name, count in sorted_results.items():
    #             f.write(f"{name}: {count}\n")
    # else:
    #     raise ValueError("save_format must be 'json' or 'txt'.")

    # print(f"统计完成，结果已保存到 {output_path}")


def json_2_txt(json_path: str, txt_path: str, num_lines: int = 100):
    """
    将 JSON 文件转换为 TXT 文件。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    with open(txt_path, 'w') as f:
        for key in list(data.keys())[:num_lines]:
            f.write(key+"\n")
    
    print(f"已将 {json_path} 转换为 {txt_path}")


if __name__ == "__main__":
    # features = torch.load('features/vgg/features.pt')
    analyze_target_similarity(
        features_path='features/clip/features.pt',
        target_txt_path='poison.txt',
        output_path='similarity_resnet.json',
        similarity_threshold=0.8,
        save_format='json'
    )
    # for n in [200,300,400,500]:
    #     txt_path=f'select_txt/resnet_{n}.txt'
    #     json_path='similarity_resnet.json'
    #     json_2_txt(json_path,txt_path,n)

    
    # print("Image 0 vs 1 cosine similarity:", sim_matrix[0, 1].item())
    # print("Image 0 vs 2 cosine similarity:", sim_matrix[0, 2].item())

