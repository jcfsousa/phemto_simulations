from collections import deque
import numpy as np
import pandas as pd

def find_neighbors(arr, index):
    x, y = arr[index]
    neighbors = np.where(
        (np.abs(arr[:, 0] - x) <= 1) &
        (np.abs(arr[:, 1] - y) <= 1)
    )[0]
    neighbors = neighbors[neighbors != index]
    return neighbors

def bfs(arr, start_index, visited):
    queue = deque([start_index])
    cluster = []

    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.add(current)
            cluster.append(current)
            neighbors = find_neighbors(arr, current)
            for n in neighbors:
                if n not in visited:
                    queue.append(n)
    return cluster

def cluster(data, global_cluster_id=1):
    """
    Cluster hits in (X,Y) separately for each (Event, Index).
    Each (Event, Index) group gets its own set of cluster IDs.
    """
    cluster_dict = {}

    # First group by Event, then by Index inside each Event
    for event_id, event_group in data.groupby('Event'):
        for idx_value, group in event_group.groupby('Index'):
            arr = group[['X', 'Y']].to_numpy()
            visited = set()

            for i in range(len(group)):
                if i not in visited:
                    cluster_local = bfs(arr, i, visited)

                    df_indices = group.index[cluster_local]
                    for df_idx in df_indices:
                        cluster_dict[df_idx] = global_cluster_id

                    global_cluster_id += 1

    return cluster_dict, global_cluster_id




df_test = pd.DataFrame({
    "Event": [1, 1, 1, 1],
    "Index": [0, 0, 1, 1],
    "X":     [10, 11, 10, 11],
    "Y":     [20, 30, 20, 20],
})

cluster_dict, next_id = cluster(df_test, global_cluster_id=1)
df_test["Cluster"] = df_test.index.map(cluster_dict)

print(df_test)
