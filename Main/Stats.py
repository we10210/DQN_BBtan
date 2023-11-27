from os.path import exists

class Stats:

    def __init__(self):
        pass

    @staticmethod
    def record_score(curr_score):
        if not curr_score:
            curr_score = 0

        file = open("highscore.txt", "w+")
        file.write(str(curr_score))
        file.close()

    @staticmethod
    def read_score():
        try:
            file = open("highscore.txt", "r")
            curr_score = file.readline().strip()
            file.close()
        except Exception as e:
            print(e)
            return 0

        if curr_score:
            return int(curr_score)
        return 0

    @staticmethod
    def calc_avg(curr_score):

        file_exists = exists('avg.txt')
        if not file_exists:
            file = open("avg.txt", "w")
            file.write('0,0')
            file.close()

        file = open("avg.txt", "r")
        line = file.readline()
        file.close()

        data = line.split(",")

        num_items = int(data[1]) if data else 0
        avg = float(data[0]) if data else 0

        total = avg * num_items
        num_items += 1
        total = (total + curr_score) / num_items

        file = open("avg.txt", "w")
        file.write(str(total) + "," + str(num_items))
        file.close()