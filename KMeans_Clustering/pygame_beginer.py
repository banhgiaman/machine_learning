import pygame
import math
from random import randint
from sklearn.cluster import KMeans

def distance(a, b):
    return math.sqrt((a[0]-b[0]) ** 2 + (a[1]-b[1]) ** 2)

pygame.init()

running = True
err = 0
K = 0
clusters = []
points = []
labels = []

screen = pygame.display.set_mode((1200,700))
clock = pygame.time.Clock()

BACKGROUND = (214, 214, 214)
BLACK = (0,0,0)
BACKGROUND_PANEL = (249, 255, 230)
WHITE = (255,255,255)

RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (147, 153, 35)
PURPLE = (255,0,255)
SKY = (0,255,255)
ORANGE = (255,125,25)
GRAPE = (100,25,125)
GRASS = (55,155,65)

COLORS = [RED, GREEN, BLUE, YELLOW, PURPLE, SKY, ORANGE, GRAPE, GRASS]

font = pygame.font.SysFont('sans', 40)
text_plus = font.render('+', True, WHITE)
text_minus = font.render('-', True, WHITE)
text_run = font.render('Run', True, WHITE)
text_random = font.render('Random', True, WHITE)
text_algorithm = font.render('Algorithm', True, WHITE)
text_reset = font.render('Reset', True, WHITE)

while running:
    clock.tick(60)
    screen.fill(BACKGROUND)
    pygame.draw.rect(screen, BLACK, pygame.Rect(50, 50, 700, 500))
    pygame.draw.rect(screen, BACKGROUND_PANEL, pygame.Rect(55, 55, 690, 490))

    mouse_x, mouse_y = pygame.mouse.get_pos()

    # +
    pygame.draw.rect(screen, BLACK, pygame.Rect(800, 50, 50, 50))
    screen.blit(text_plus, (820, 50))

    # -
    pygame.draw.rect(screen, BLACK, pygame.Rect(900, 50, 50, 50))
    screen.blit(text_minus, (920, 50))

    # K
    text_K = font.render('K = ' + str(K), True, BLACK)
    screen.blit(text_K, (970, 50))

    # run
    pygame.draw.rect(screen, BLACK, pygame.Rect(800, 150, 200, 50))
    screen.blit(text_run, (820, 150))

    # random
    pygame.draw.rect(screen, BLACK, pygame.Rect(800, 250, 200, 50))
    screen.blit(text_random, (820, 250))

    # error
    text_error = font.render('Error = ' + str(int(err)) , True, BLACK)
    screen.blit(text_error, (820, 350))

    # alg
    pygame.draw.rect(screen, BLACK, pygame.Rect(800, 450, 200, 50))
    screen.blit(text_algorithm, (820, 450))

    # reset 
    pygame.draw.rect(screen, BLACK, pygame.Rect(800, 550, 200, 50))
    screen.blit(text_reset, (820, 550))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            # add new point
            if 50 < mouse_x < 750 and 50 < mouse_y < 550:
                labels = []
                clusters = []
                points.append((mouse_x, mouse_y))

            # +
            if 800 < mouse_x < 850 and 50 < mouse_y < 100:
                if K < 8:
                    K += 1

            # -
            if 900 < mouse_x < 950 and 50 < mouse_y < 100:
                if K > 0:
                    K -= 1

            # run
            if 800 < mouse_x < 1000 and 150 < mouse_y < 200:
                if len(clusters) == 0:
                    continue

                labels = []
                points_category = []
                for i in range(K):
                    points_category.append([])
                for p in points:
                    distances_to_cluster = []
                    for c in clusters:
                        dis = distance(p, c)
                        distances_to_cluster.append(dis)
                    min_distance = min(distances_to_cluster)
                    label = distances_to_cluster.index(min_distance)
                    points_category[label].append(p)
                    labels.append(label)
                for i in range(K):
                    if len(points_category[i]):
                        sum_x = 0
                        sum_y = 0
                        for p_c in points_category[i]:
                            sum_x += p_c[0]
                            sum_y += p_c[1]
                        new_x = sum_x / len(points_category[i])
                        new_y = sum_y / len(points_category[i])
                        clusters[i] = (new_x, new_y)

                # Calculate error
                err = 0
                for i in range(len(points)):
                    err += distance(points[i], clusters[labels[i]])

            # random
            if 800 < mouse_x < 1000 and 250 < mouse_y < 300:
                labels = []
                clusters = []
                for i in range(K):
                    random_point = (randint(50, 750), randint(50, 550))
                    clusters.append(random_point)

            # algorithm
            if 800 < mouse_x < 1000 and 450 < mouse_y < 500:
                try:
                    kmeans = KMeans(n_clusters=K).fit(points)
                    labels = kmeans.labels_
                    clusters = kmeans.cluster_centers_
                    # Calculate error
                    err = 0
                    for i in range(len(points)):
                        err += distance(points[i], clusters[labels[i]])
                except:
                    print('error')

            if 800 < mouse_x < 1000 and 550 < mouse_y < 600:
                points = []
                labels = []
                clusters = []
                err = 0
                K = 0

    for i in range(len(clusters)):
        pygame.draw.circle(screen, COLORS[i], clusters[i], (10))

    for i in range(len(points)):
        pygame.draw.circle(screen, BLACK , points[i], (6))
        if len(labels) > 0:
            pygame.draw.circle(screen, COLORS[labels[i]] , points[i], (5))
        else:
            pygame.draw.circle(screen, WHITE , points[i], (5))
    
    pygame.display.flip()

pygame.quit()