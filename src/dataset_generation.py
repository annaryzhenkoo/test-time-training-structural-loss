import random
import pandas as pd

def dataset_generation(dataset_size: int, mode: str, num_digits: int):
    dataset = []

    low = 0 if num_digits == 3 else 10 ** (num_digits - 1)
    high = 10 ** num_digits - 1

    for _ in range(dataset_size):
        a_num = random.randint(low, high)
        b_num = random.randint(low, high)
        sum_num = a_num + b_num

        if mode == "binary":
            a = bin(a_num)[2:]
            a = a[::-1]
            b = bin(b_num)[2:]
            b = b[::-1]
            result = bin(sum_num)[2:]
            result = result[::-1] #reverse
        else:
            a = str(a_num)
            b = str(b_num)
            result = str(sum_num)

        addition = f"{a}+{b}={result}"
        dataset.append(addition)

    df = pd.DataFrame(dataset)

    print("Dataset size:", len(df))

    df.to_csv(
        f"data/data_dn{num_digits}_ds{dataset_size}_m{mode}.csv",
        index=False
    )

    return f"data/data_dn{num_digits}_ds{dataset_size}_m{mode}.csv"
