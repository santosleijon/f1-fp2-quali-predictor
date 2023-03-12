from get_traning_set_for_rounds import get_training_set_for_rounds 

training_set = get_training_set_for_rounds(18,50)

training_set.to_csv('traning_set.csv', encoding='utf-8', index=False)
