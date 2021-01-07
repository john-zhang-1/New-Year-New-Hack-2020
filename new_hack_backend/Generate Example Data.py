import csv
import random

variables = ['Meal', 'Bedtime', 'Waketime', 'Quality', 'Electronics', 'Up', 'Temperature', 'Noise', 'Nap']

standards = [120, 1380, 480, 0, 15, 1, 23, 30, 15]
noise_max = [60, 240, 240, 0, 15, 2, 4, 30, 15]
weighting = [0.5, 1.5, 1.5, 0, 0.5, 1, 1, 1, 1]

def create_doc():
    with open('Example Model/random_gen_data.csv', mode = 'w') as myfile:

        quality_ind = variables.index('Quality')
        up_ind = variables.index('Up')
        bedtime_ind = variables.index('Bedtime')

        writer = csv.writer(myfile)
        writer.writerow(variables)

        for day in range(100):

            new_point = []
            for i in standards:
                new_point.append(i)

            for i in range(len(variables)):
                noise = random.random() * noise_max[i]

                if random.random() > 0.5:
                    new_point[i] += noise
                else:
                    new_point[i] -= noise

                if i == bedtime_ind and new_point[bedtime_ind] >= 1440:
                    new_point[bedtime_ind] -= 1440

            new_point[up_ind] = round(abs(new_point[up_ind]))

            for i in range(len(variables)):
                if i != quality_ind:
                    new_point[quality_ind] += 2.25 * random.random() * (1 - abs((standards[i] - new_point[i])/standards[i])) * weighting[i]

            if new_point[quality_ind] > 10:
                new_point[quality_ind] = 10
            new_point[quality_ind] = round(new_point[quality_ind])

            writer.writerow(new_point)

create_doc()