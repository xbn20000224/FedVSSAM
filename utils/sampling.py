import numpy as np












def cifar10_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar10_noniid(args, dataset, num_users):
    alpha = args.alpha
    labels = np.array(dataset.targets)
    n_client = num_users
    n_cls = int(labels.max() + 1)


    rng = np.random.default_rng(getattr(args, "seed", None))
    idx_list = [np.where(labels == c)[0] for c in range(n_cls)]
    for c in range(n_cls):
        rng.shuffle(idx_list[c])


    n_total = len(labels)
    n_data_per_client = int(n_total / n_client)
    client_data_list = np.ones(n_client, dtype=int) * n_data_per_client
    diff = int(client_data_list.sum() - n_total)
    if diff != 0:

        for u in range(n_client):
            if client_data_list[u] > diff:
                client_data_list[u] -= diff
                break



    cls_priors = rng.dirichlet(alpha * np.ones(n_cls), size=n_client)  # shape: [n_client, n_cls]
    prior_cumsum = np.cumsum(cls_priors, axis=1)


    cls_amount = [len(idx_list[c]) for c in range(n_cls)]


    dict_users = {u: [] for u in range(n_client)}


    while np.sum(client_data_list) != 0:
        curr_client = rng.integers(0, n_client)
        if client_data_list[curr_client] <= 0:
            continue


        client_data_list[curr_client] -= 1


        u = rng.random()
        cls_label = int(np.searchsorted(prior_cumsum[curr_client], u, side="right"))


        if cls_amount[cls_label] <= 0:
            cls_amount[cls_label] = len(idx_list[cls_label])


        cls_amount[cls_label] -= 1
        take_idx = idx_list[cls_label][cls_amount[cls_label]]

        dict_users[curr_client].append(int(take_idx))


    for u in range(n_client):
        arr = np.array(dict_users[u], dtype=np.int64)
        rng.shuffle(arr)
        dict_users[u] = arr

    return dict_users


def cifar10_pathological(args, dataset, num_users):
    """
    Create pathological non-IID data distribution.

    Each user is assigned `num_classes_per_user` classes, and receives only data from those classes.
    Typically, num_classes_per_user=2 mimics the FedAvg paper setting.

    Args:
        dataset: CIFAR-10 Dataset (with `targets` attribute).
        num_users: Number of clients.
        num_classes_per_user: How many classes each client gets.

    Returns:
        dict_users: dict mapping user_id -> list of sample indices.
    """
    num_classes_per_user = args.num_classes_per_user
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))  # Should be 10 for CIFAR-10
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # Step 1: Group sample indices by class
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}

    # Step 2: Shuffle each class index list
    for c in class_indices:
        np.random.shuffle(class_indices[c])

    # Step 3: Compute total shards = num_users * num_classes_per_user
    shards_per_class = (num_users * num_classes_per_user) // num_classes
    assert shards_per_class > 0, "Too few users or too many classes per user"

    # Step 4: Split each class into shards
    shards = []
    for c in range(num_classes):
        class_idxs = class_indices[c]
        shard_size = len(class_idxs) // shards_per_class
        for i in range(shards_per_class):
            shard = class_idxs[i * shard_size: (i + 1) * shard_size]
            shards.append((c, shard))  # Keep track of class

    np.random.shuffle(shards)

    # Step 5: Assign shards to users
    user_shard_counts = {i: 0 for i in range(num_users)}
    for user in range(num_users):
        assigned_classes = set()
        while user_shard_counts[user] < num_classes_per_user:
            for i, (cls, shard) in enumerate(shards):
                if cls not in assigned_classes:
                    dict_users[user] = np.concatenate((dict_users[user], shard), axis=0)
                    assigned_classes.add(cls)
                    user_shard_counts[user] += 1
                    del shards[i]
                    break
            else:

                break

    return dict_users






def cifar100_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar100_noniid(args, dataset, num_users):
    alpha = args.alpha
    labels = np.array(dataset.targets)
    n_client = num_users
    n_cls = int(labels.max() + 1)
    print("num of classes:", n_cls)


    rng = np.random.default_rng(getattr(args, "seed", None))
    idx_list = [np.where(labels == c)[0] for c in range(n_cls)]
    for c in range(n_cls):
        rng.shuffle(idx_list[c])


    n_total = len(labels)
    n_data_per_client = int(n_total / n_client)
    client_data_list = np.ones(n_client, dtype=int) * n_data_per_client
    diff = int(client_data_list.sum() - n_total)
    if diff != 0:

        for u in range(n_client):
            if client_data_list[u] > diff:
                client_data_list[u] -= diff
                break



    cls_priors = rng.dirichlet(alpha * np.ones(n_cls), size=n_client)  # shape: [n_client, n_cls]
    prior_cumsum = np.cumsum(cls_priors, axis=1)


    cls_amount = [len(idx_list[c]) for c in range(n_cls)]


    dict_users = {u: [] for u in range(n_client)}


    while np.sum(client_data_list) != 0:
        curr_client = rng.integers(0, n_client)
        if client_data_list[curr_client] <= 0:
            continue


        client_data_list[curr_client] -= 1


        u = rng.random()
        cls_label = int(np.searchsorted(prior_cumsum[curr_client], u, side="right"))


        if cls_amount[cls_label] <= 0:
            cls_amount[cls_label] = len(idx_list[cls_label])


        cls_amount[cls_label] -= 1
        take_idx = idx_list[cls_label][cls_amount[cls_label]]

        dict_users[curr_client].append(int(take_idx))


    for u in range(n_client):
        arr = np.array(dict_users[u], dtype=np.int64)
        rng.shuffle(arr)
        dict_users[u] = arr

    return dict_users


def cifar100_pathological(args, dataset, num_users):
    """
    Create pathological non-IID data distribution.

    Each user is assigned `num_classes_per_user` classes, and receives only data from those classes.
    Typically, num_classes_per_user=2 mimics the FedAvg paper setting.

    Args:
        dataset: CIFAR-10 Dataset (with `targets` attribute).
        num_users: Number of clients.
        num_classes_per_user: How many classes each client gets.

    Returns:
        dict_users: dict mapping user_id -> list of sample indices.
    """
    num_classes_per_user = args.num_classes_per_user
    labels = np.array(dataset.targets)
    num_classes = 100
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # Step 1: Group sample indices by class
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}

    # Step 2: Shuffle each class index list
    for c in class_indices:
        np.random.shuffle(class_indices[c])

    # Step 3: Compute total shards = num_users * num_classes_per_user
    shards_per_class = (num_users * num_classes_per_user) // num_classes
    assert shards_per_class > 0, "Too few users or too many classes per user"

    # Step 4: Split each class into shards
    shards = []
    for c in range(num_classes):
        class_idxs = class_indices[c]
        shard_size = len(class_idxs) // shards_per_class
        for i in range(shards_per_class):
            shard = class_idxs[i * shard_size: (i + 1) * shard_size]
            shards.append((c, shard))  # Keep track of class

    np.random.shuffle(shards)

    # Step 5: Assign shards to users
    user_shard_counts = {i: 0 for i in range(num_users)}
    for user in range(num_users):
        assigned_classes = set()
        while user_shard_counts[user] < num_classes_per_user:
            for i, (cls, shard) in enumerate(shards):
                if cls not in assigned_classes:
                    dict_users[user] = np.concatenate((dict_users[user], shard), axis=0)
                    assigned_classes.add(cls)
                    user_shard_counts[user] += 1
                    del shards[i]
                    break
            else:

                break

    return dict_users
