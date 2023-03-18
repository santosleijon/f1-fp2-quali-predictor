from get_data_set import get_data_set

test_set = get_data_set(1, 44)

test_set.to_csv('data_set_2021-2022.csv', encoding='utf-8', index=False)
