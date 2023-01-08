import os
import random


def split_trainval_test(input_dir, rate=0.6):
    all_file = []
    for i, cls_name in enumerate(os.listdir(input_dir)):
        file_dir = os.path.join(input_dir, cls_name, "images")
        for img_name in os.listdir(file_dir):
            if img_name.endswith(".png"):
                img_path = os.path.join(file_dir, img_name)
                all_file.append([img_path, str(i)])  # 图片路径和标签

    random.shuffle(all_file)
    train = all_file[: int(len(all_file) * rate)]
    val = all_file[int(len(all_file) * rate): int(len(all_file) * (1+rate)/2)]
    test = all_file[int(len(all_file) * (1+rate)/2):]

    return train, val, test


def generate_train_val_test_txt_file(train, val, test, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    train_txt_path = os.path.join(save_dir, "train.txt")
    val_txt_path = os.path.join(save_dir, "val.txt")
    test_txt_path = os.path.join(save_dir, "test.txt")

    train_str = ""
    val_str = ""
    test_str = ""

    for img_path, cls_name in train:
        train_str += img_path + "\t" + cls_name + "\n"
    for img_path, cls_name in val:
        val_str += img_path + "\t" + cls_name + "\n"
    for img_path, cls_name in test:
        test_str += img_path + "\t" + cls_name + "\n"
    with open(train_txt_path, "w") as fw:
        fw.write(train_str)
    with open(val_txt_path, "w") as fw:
        fw.write(val_str)
    with open(test_txt_path, "w") as fw:
        fw.write(test_str)


if __name__ == "__main__":
    input_dir = r"/home/rain/PythonProject/GFD/data/gearbox/img".replace("\\", "/")
    save_dir = input_dir
    train, val, test = split_trainval_test(input_dir, rate=0.6)
    generate_train_val_test_txt_file(train, val, test, save_dir)
