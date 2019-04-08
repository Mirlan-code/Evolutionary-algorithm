from random import randrange
import numpy
import copy
from PIL import Image

# 512 * 512 = 512/16 * 512/16
block_size = 16
n = int(512 / block_size)
population = 10
image = Image.open('kek.jpg')
input_image = numpy.array(image)
current_generation = numpy.zeros([n, n, population, block_size, block_size, 3])
avg = [0, 0, 0]
for index in numpy.ndindex(n, n, population):
    block_row, block_col, population_number = index[0], index[1], index[2]
    total = block_size * block_size
    for index2 in numpy.ndindex(block_size, block_size):
        i, j = index2[0], index2[1]
        r, g, b = input_image[block_row * block_size + i, block_col * block_size + j]
        avg[0] += r
        avg[1] += g
        avg[2] += b
    avg[0] /= total
    avg[1] /= total
    avg[2] /= total
    for index2 in numpy.ndindex(block_size, block_size):
        i, j = index2[0], index2[1]
        r, g, b = avg
        r += randrange(-40, 40)
        r = max(r, 0)
        r = min(r, 255)
        g += randrange(-40, 40)
        g = max(g, 0)
        g = min(g, 255)
        b += randrange(-40, 40)
        b = max(b, 0)
        b = min(b, 255)
        current_generation[block_row, block_col, population_number, i, j] = [r, g, b]

print("Here we go")


# does crossover between two generations with some deviations
def crossover(mother, father, goal):
    child = numpy.zeros([block_size, block_size, 3])

    for index in numpy.ndindex(block_size, block_size):
        i, j = index[0], index[1]
        m = mother[i, j]
        f = father[i, j]
        for k in range(3):
            mid = randrange(int(min(m[k], f[k])), int(max(m[k], f[k])) + 1)
            if mid < goal[i, j, k]:
                mid = mid + 1
            elif mid > goal[i, j, k]:
                mid = mid - 2
            child[i, j, k] = mid

    return child


# calculates the similarity of the image - the Manhattan distance to the average color in the block
def fitness_similar(a: Image):
    differenceA = 0
    avgA = [0, 0, 0]
    total = block_size * block_size

    for index in numpy.ndindex(block_size, block_size):
        i, j = index[0], index[1]
        r, g, b = a[i, j]
        avgA[0] += r / total
        avgA[1] += g / total
        avgA[2] += b / total
    for index in numpy.ndindex(block_size, block_size):
        i, j = index[0], index[1]
        r, g, b = a[i, j]
        differenceA += abs(avgA[0] - r) + abs(avgA[1] - g) + abs(avgA[2] - b)
    all_world = 255 * 3 * total
    return 1 - differenceA / all_world


# chooses the better image according to the similarity of colors
def better(a: Image, b: Image):
    if fitness_similar(a) > fitness_similar(b):
        return True
    else:
        return False


# random mutation of the generation
def mutation(a: Image):
    res = a
    for index in numpy.ndindex(block_size, block_size):
        i, j = index[0], index[1]
        r, g, b = a[i, j]
        r += randrange(-100, 100)
        r = max(r, 0)
        r = min(r, 255)
        g += randrange(-100, 100)
        g = max(g, 0)
        g = min(g, 255)
        b += randrange(-100, 100)
        b = max(b, 0)
        b = min(b, 255)
        res[i, j] = r, g, b

    return res;


def update(block_row, block_col):
    golden_number = 0.93
    image_part = numpy.zeros([block_size, block_size, 3])
    for index in numpy.ndindex(block_size, block_size):
        i, j = index[0], index[1]
        image_part[i, j] = input_image[block_row * block_size + i, block_col * block_size + j]

    block_generation = current_generation[block_row][block_col]

    # sort them from the best to the worst
    print(str(fitness_similar(block_generation[0])))
    if fitness_similar(block_generation[0]) > golden_number:
        return False

    for i in range(population):
        for j in range(i + 1, population):
            if better(block_generation[j - 1], block_generation[j]) == False:
                tmp = copy.deepcopy(block_generation[j])
                block_generation[j] = block_generation[j - 1]
                block_generation[j - 1] = tmp
    # first half stays alive, second die
    if (randrange(0, 5) == 0):
        print("mutation is going")
        block_generation[4] = mutation(block_generation[4])
    for child in range(int(population / 2), population):
        # picks mother and father from the first half
        mother = randrange(0, population / 2)
        father = randrange(0, population / 2)
        block_generation[child] = crossover(block_generation[mother], block_generation[father], image_part)


# calculates the Manhattan distance to the ideal image
def fitness_rect(rect: Image, goal: Image, x1, y1, x2, y2):
    diff = 0
    total = (x2 - x1 + 1) * (y2 - y1 + 1)
    for i in range(x1, x2 + 1):
        for j in range(y1, y2 + 1):
            ar, ag, ab = rect[i, j]
            gr, gg, gb = goal[i, j]
            diff += abs(ar - gr) + abs(ag - gg) + abs(ab - gb)
    all_world = 255 * 3 * total
    return 1 - diff/all_world


# choose the closest one to the ideal
def better_rect(a: Image, b: Image, goal: Image, x1, y1, x2, y2):
    if fitness_rect(a, goal, x1, y1, x2, y2) > fitness_rect(b, goal, x1, y1, x2, y2):
        return True
    else:
        return False


# updates rect (does new generation)
def update_rect(block_row, block_col, x1, y1, x2, y2):
    golden_number = 0.93  # Manhattan distance between ideal and ours
    image_part = numpy.zeros([block_size, block_size, 3])
    for index in numpy.ndindex(block_size, block_size):
        i, j = index[0], index[1]
        image_part[i, j] = input_image[block_row * block_size + i, block_col * block_size + j]

    block_generation = current_generation[block_row, block_col]
    print(str(fitness_rect(block_generation[0], image_part, x1, y1, x2, y2)))
    if fitness_rect(block_generation[0], image_part, x1, y1, x2, y2) > golden_number:
        return False
    # sorts the current population
    for i in range(population):
        for j in range(i + 1, population):
            if better_rect(block_generation[j - 1], block_generation[j], image_part, x1, y1, x2, y2) == False:
                tmp = copy.deepcopy(block_generation[j])
                block_generation[j] = copy.deepcopy(block_generation[j - 1])
                block_generation[j - 1] = copy.deepcopy(tmp)
    # first half stays alive, second die
    if(randrange(0, 5) == 0):
        print("mutation is going")
        block_generation[4] = mutation(block_generation[4])
    for child in range(int(population / 2), population):
        # picks mother and father from the first half
        mother = randrange(0, population / 2)
        father = randrange(0, population / 2)
        block_generation[child] = crossover(block_generation[mother], block_generation[father], image_part)
    return True


gen = 0

generations_rect = 0

# blocks generation to the ideal image update_rect
for index in numpy.ndindex(n, n):
    block_row, block_col = index[0], index[1]
    x = min(block_col, block_row, n - block_row, n - block_col)
    x = int(x)
    for gen_rect in range(generations_rect):
        print("generation number is " + str(gen) + " " + str(block_row) + " " + str(block_col))
        gen = gen + 1
        if update_rect(block_row, block_col, 0, 0, block_size - 1, block_size - 1) == False:
            break

generations = 2000
generations_rect = 2

# random rectangle to the ideal image update_rect
for index in numpy.ndindex(generations):
    _, block_row, block_col = index[0], randrange(0, n), randrange(0, n)
    x1, y1, x2, y2 = randrange(0, block_size / 2), randrange(0, block_size / 2), randrange(block_size / 2,
                                                                                           block_size), randrange(
        block_size / 2, block_size)
    if (x1 > x2):
        x1, x2 = x2, x1
    if (y1 > y2):
        y1, y2 = y2, y1

    # block_row, block_col, x1, y1, x2, y2 = 0, 0, 0, 0, block_size - 1, block_size - 1
    print("rectangle is " + str(block_row * block_size + x1) + " " + str(block_col * block_size + y1) + " " + str(
        block_row * block_size + x2) + " " + str(block_col * block_size + y2));
    for index2 in numpy.ndindex(generations_rect):
        gen_rect = index2[0]
        print("generation number is " + str(gen))
        gen = gen + 1
        if update_rect(block_row, block_col, x1, y1, x2, y2) == False:
            break

generations_rect = 1
# similar update
for block_row in range(n):
    for block_col in range(n):
        for t in range(generations_rect):
            print("generation number is " + str(gen))
            gen = gen + 1
            update(block_row, block_col)

final_result = input_image
best = 0
for index in numpy.ndindex(n, n, block_size, block_size):
    block_row, block_col, i, j = index[0], index[1], index[2], index[3]

    final_result[block_row * block_size + i, block_col * block_size + j] = (
        current_generation[block_row, block_col, best, i, j])

print("Congrats")

final_image = Image.fromarray(final_result)
final_image.show()
final_image.save("res.jpg")
