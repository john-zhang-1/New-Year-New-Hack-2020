import csv
import random

variables = ['Meal', 'Bedtime', 'Waketime', 'Length', 'Quality', 'Electronics', 'Up', 'Temperature', 'Noise', 'Nap']

standards = [120, 1380, 480, 540, 0, 15, 1, 23, 30, 15]
noise_max = [60, 240, 240, 0, 0, 15, 2, 4, 30, 15]
weighting = [0.5, 1.5, 1.5, 3, 0, 0.5, 1, 1, 1, 1]

def create_doc():
    with open('random_gen_data.csv', mode = 'w') as myfile:
        writer = csv.writer(myfile)
        writer.writerow(variables)

        for i in range(100):

            new_point = []
            for i in standards:
                new_point.append(i)

            for i in range(10):
                noise = random.random() * noise_max[i]

                if random.random() > 0.5:
                    new_point[i] += noise
                else:
                    new_point[i] -= noise

                if i == 1 and new_point[1] >= 1440:
                    new_point[1] -= 1440

            new_point[6] = round(abs(new_point[6]))
            new_point[3] = new_point[2] - new_point[1] if new_point[2] >= new_point[1] else new_point[2] - new_point[1] + 1440
            new_point[4] = 0
            for i in range(len(variables)):
                if i != 4:
                    new_point[4] += 1.75 * random.random() * (1 - abs(standards[i] - new_point[i])/standards[i]) * weighting[i]

            if new_point[4] > 10:
                new_point[4] = 10
            new_point[4] = round(new_point[4])

            writer.writerow(new_point)

if __name__ == '__main__':
    create_doc()