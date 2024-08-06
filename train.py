import numpy as np
import re
from SMO import SMO_GAUSSIAN
import concurrent.futures
import os
import time

# data len
data_train_len = 5_000
#data_test_len  = 1_000

# paths
base_dir = "./dataset"
initial_words_path = os.path.join(base_dir, "google-10000-english.txt")
stop_words_path = os.path.join(base_dir, "stop_words_english.txt")
storage_path = os.path.join(base_dir, "parameters.npz")

# for words to consider
max_words_to_consider: int = 15_000
max_word_len:          int = 15
words_to_consider: dict = {}
stop_words:        dict = {}

# for pairs
train_path:        str = "train/"
test_path:         str = "test/"
c:                 str = ["anger", "fear", "joy", "sadness"]
total_class:       int = len(c)
total_pairs:       int = int((total_class * (total_class - 1))/2)

# for vectorized training data
v_train: np.int32 = np.zeros(shape=(total_class, data_train_len, max_words_to_consider))
#v_test:  np.int32 = np.zeros(shape=(total_class, data_test_len, max_words_to_consider))


# Responsible for parsing 
def parse_initial_words(destination: dict, path: str, add_undefined_token: bool=False, filter_by: dict = {}, limit_len: int = -1) -> None:
    if add_undefined_token:
        destination["$$$"] = 0
    
    print("Parsing words")
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            word: str = line.strip()
            if (limit_len > -1 and len(destination) == limit_len):
                continue
            
            if (len(word) <= 1 or len(word) >= max_word_len):
                continue
            
            if (word in filter_by):
                continue

            destination[word] = len(destination)

    print("parsing succesfull")


# Checks text word by word. If a word not in word_to_consider then we add it
# unless we already exceeded the max words to consider, or the word is not a
# bad candidate. We then update the occurence of the word in its corresponding matrix
def process_text(destination: np.int32, line: str, i: int, line_idx: int, add_new:bool=True):
    D: int= len(words_to_consider)

    for word in line.split():
        word: str = re.sub(r'[^a-zA-Z]+$', '', word)

        if word not in words_to_consider:
            if len(word) <= 1 or len(word) >= max_word_len:
                continue

            if D >= max_words_to_consider:
                continue

            if not add_new:
                continue
             
            words_to_consider[word] = D
            D += 1

        destination[i, line_idx, words_to_consider[word]] += 1


def conv_data_to_vectors(destination: np.int32, source_path: str, data_len: np.int32, name: str, add_new: bool=True):
    print(f'Converting {name} data to vectors...')
    for i in range(total_class):
        data_path: str = source_path + c[i] + ".txt"
        
        with open(data_path, 'r', encoding='utf-8') as file:
            line_idx: int = 0
            for line in file:
                if (line_idx == data_len):
                    break
                
                process_text(destination, line, i, line_idx, add_new)
                line_idx += 1


# A mapping technique such that we can store our class_pairs in a 1 dimensional array. 
def map_to_pos(size: int, i: int, j: int):
    return ((2*size - i - 3)*i + 2*(j - 1))/2


if __name__ == "__main__":
    # ====> Parse the words to consider
    parse_initial_words(stop_words, stop_words_path, False)
    #parse_initial_words(words_to_consider, initial_words_path, True, stop_words, max_words_to_consider)


    # ====> convert data to vectors 
    conv_data_to_vectors(v_train, train_path, data_train_len, "train")
    #conv_data_to_vectors(v_test, test_path, data_test_len, "test", add_new=False)

    print("Conversion complete...")
    print("Total words to consider: ", len(words_to_consider))


    # ====> train dataset 
    start = time.time()
    class_pairs: list = []
    target: np.float64 = np.concatenate((np.full(shape=data_train_len, fill_value=1), np.full(shape=data_train_len, fill_value=-1)), axis=0)

    # Train in Parallel (division at a time)
    # divisions should divide total classes to int
    alpha_res: np.float64 = np.zeros(shape=(total_pairs, data_train_len*2), dtype=np.float64)
    beta_res : np.float64 = np.zeros(shape=(total_pairs), dtype=np.float64)

    divisions: int = 2
    chunk:     int = int(total_pairs/divisions)
    
    for i in range(divisions):
        # initialize the classes chunk at a time
        for _ in range(chunk):
            class_pairs.append(SMO_GAUSSIAN(np.concatenate((v_train[0], v_train[1]), axis=0),
                                            target, data_train_len*2, max_words_to_consider, c=.7, log=False))
            
        with concurrent.futures.ProcessPoolExecutor() as executor:
            processes = [executor.submit(class_pairs[i].smo_train, i) for i in range(chunk*i, chunk*(i + 1))]

            for f in concurrent.futures.as_completed(processes):
                idx, alphs, B = f.result()
                alpha_res[idx], beta_res[idx] = alphs, B
                print("Finished training for pair: ", idx)
    
    print("Total Runtime: ", time.time() - start)
    
    # ====> save
    print("saving parameters...")
    np.savez_compressed(storage_path, alpha_pairs=alpha_res, betas=beta_res)
    print("save complete")
