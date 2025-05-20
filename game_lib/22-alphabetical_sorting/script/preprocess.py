from tqdm import tqdm

if __name__ == "__main__":

    words = []
    words_set = set()
    with open("ori_words.txt", "r") as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            word = line.split(" ")[0]
            # 过滤掉长度为2的单词
            flag = True
            if len(word)<=2:
                flag=False
            for c in word.lower():
                if not ord('a') <= ord(c) <= ord('z'):
                    flag = False
                    break
            if flag and word not in words_set:
                words.append(word.lower())
                words_set.add(word.lower())
    print(len(words))
    with open("words.txt", "w") as f:
        for word in words:
            f.write(word + "\n")
    