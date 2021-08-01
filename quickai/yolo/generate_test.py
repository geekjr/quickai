import os


def generate_test(path):
    image_files = []
    os.chdir(os.path.join("data", "test"))
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".jpg"):
            image_files.append(path + filename)
    os.chdir("..")
    with open("test.txt", "w") as outfile:
        for image in image_files:
            outfile.write(image)
            outfile.write("\n")
        outfile.close()
    os.chdir("..")
