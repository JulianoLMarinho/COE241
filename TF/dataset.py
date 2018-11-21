class Dataset:
    data = {"headers": [], "data": []}
    def __init__(self, path_to_csv):
        file_object = open(path_to_csv, "r")
        self.create_list(file_object)


    def create_list(self, fileObejct):
        lines = fileObejct.readlines()
        for i in range(len(lines)):
            if i == 0:
                self.data["headers"] = lines[i].split()
            else:
                self.data["data"].append(lines[i].split())

    def describe(self):
        media = {}
        for item in self.data["data"]:
            media += int(item[0])

        return media*1.0/len(self.data["data"])