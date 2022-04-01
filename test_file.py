print("If this file executes correct, it will create and save a simple text file with some numbers.\n")

l = [1, 2, 3, 4, 5, 6]


with open('test_txt_file.txt', 'w') as f:
    for nr in l:
        f.write(str(nr))

print("execution finished")
