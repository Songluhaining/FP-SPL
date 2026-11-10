import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score


def flatten_failed_traces(failed_trace):
    """
    把 failed_trace 这个 dict-of-dict 展开成矩阵和样本 ID 列表。

    Parameters
    ----------
    failed_trace : dict
        {
          variant_name: {
             coverage_file_name: [0/1 vector],
             ...
          },
          ...
        }

    Returns
    -------
    X : np.ndarray, shape (n_traces, n_stmts)
        所有失败用例的覆盖向量矩阵
    ids : list of (variant, coverage_file)
        与 X 一一对应的样本 ID
    """
    X = []
    ids = []
    for variant, test_dict in failed_trace.items():
        for cf, vec in test_dict.items():
            X.append(vec)
            ids.append((variant, cf))

    if len(X) == 0:
        raise ValueError("failed_trace 里没有任何失败用例，无法聚类。")

    X = np.asarray(X, dtype=np.float32)
    return X, ids


def cluster_failed_traces(
    failed_trace,
    min_k=2,
    max_k=3,
    random_state=0,
):
    """
    对所有未通过测试用例对应的 trace 进行聚类。

    思路：
    - 每个 trace 是一个 0/1 覆盖向量；
    - 先做一层“TF-IDF 化”加权：稀有但区分度高的语句权重大，所有/几乎所有 trace 都覆盖的语句权重小；
    - 再做一层列过滤：丢弃从不被覆盖或所有 trace 都覆盖的语句；
    - 之后做 L2 归一化，再用 KMeans 聚类（等价于在余弦相似度下做聚类）；
    - 在 k ∈ [min_k, max_k] 上用 silhouette score 自动选最优簇数。

    Parameters
    ----------
    failed_trace : dict
        外层 key 为 variant 名，内层 key 为覆盖文件名，value 为 0/1 向量。
    min_k : int
        最少聚成多少个簇。
    max_k : int
        最多聚成多少个簇。
    random_state : int
        随机种子，保证可复现。

    Returns
    -------
    best_k : int
        选出来的最优簇数。
    labels : np.ndarray, shape (n_traces,)
        每个 trace 的簇标签（0..best_k-1）。
    cluster_members : dict
        {cluster_id: [(variant, coverage_file), ...]}，便于你查看每个簇包含哪些失败用例。
    best_score : float
        最优 silhouette score，便于后续判断“其实是不是 1 个簇”。
    """
    X, ids = flatten_failed_traces(failed_trace)
    n_samples = X.shape[0]

    # 只有 1 个样本的极端情况：直接视为 1 个簇
    if n_samples == 1:
        labels = np.zeros(1, dtype=int)
        cluster_members = {0: [ids[0]]}
        return 1, labels, cluster_members, 0.0

    # ====== 2.2 列过滤：去掉“从不覆盖”或“所有 trace 都覆盖”的语句 ======
    # df_j = 有多少条 trace 覆盖了第 j 个语句
    df = X.sum(axis=0)                      # shape: (n_features,)
    # 只保留 0 < df < n_samples 的语句（至少被 1 条覆盖，但不是所有 trace 都覆盖）
    valid_mask = (df > 0) & (df < n_samples)

    if valid_mask.any():
        X = X[:, valid_mask]
        df = df[valid_mask]
    else:
        # 极端情况：所有语句要么全 0 要么全 1，那就用原始 X（说明样本本身就几乎不可分）
        df = X.sum(axis=0)

    # ====== 2.1 TF-IDF 化：对“稀有但具有区分度”的语句提高权重 ======
    # idf_j = log((N + 1) / (df_j + 1)) + 1   （+1 平滑，避免 0）
    # 直觉：df 越小（稀有语句），idf 越大；df 越接近 N，idf 越小
    idf = np.log((n_samples + 1.0) / (df + 1.0)) + 1.0      # shape: (n_features_filtered,)

    # 广播相乘：每条 trace 的该语句位乘以对应 idf
    X_weighted = X * idf  # shape: (n_samples, n_features_filtered)

    # 有效的最大簇数不能超过样本数
    max_k_eff = min(max_k, n_samples)
    if max_k_eff < 2:
        max_k_eff = 2

    # L2 归一化后，用欧氏距离的 KMeans 等价于在余弦距离下做聚类
    X_norm = normalize(X_weighted, norm="l2", axis=1)

    best_k = None
    best_score = -1.0
    best_labels = None

    for k in range(max(min_k, 2), max_k_eff + 1):
        kmeans = KMeans(
            n_clusters=k,
            n_init=20,          # 多次随机初始化，稳定一些
            max_iter=300,
            random_state=random_state,
        )
        labels = kmeans.fit_predict(X_norm)

        # silhouette_score 要求每个簇至少有 2 个样本，否则会报错或返回无意义结果
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2:
            # 没法算 silhouette，直接跳过这个 k
            # print(f"[cluster] skip k={k}: only one cluster found, counts={counts}")
            continue
        if np.any(counts < 2):
            # 如果有簇只有 1 个样本，silhouette 没有太大意义，跳过这个 k
            continue

        score = silhouette_score(X_norm, labels, metric="cosine")
        # print(f"k={k}, silhouette_score={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    # 如果上面循环里因为各种原因没有得到 best_labels（极端情况），
    # 退而求其次，直接用 k=2 的结果。
    if best_labels is None:
        kmeans = KMeans(
            n_clusters=2,
            n_init=20,
            max_iter=300,
            random_state=random_state,
        )
        best_labels = kmeans.fit_predict(X_norm)
        best_k = 2
        # 这里的 best_score 就保留为初始值 -1.0，后面你可以据此判断“极端情况”

    # 汇总每个簇里有哪些 (variant, coverage_file)
    cluster_members = defaultdict(list)
    for (variant, cf), label in zip(ids, best_labels):
        cluster_members[int(label)].append((variant, cf))

    return best_k, best_labels, dict(cluster_members), best_score
