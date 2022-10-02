import pygame
import sys
from random import randrange
import time
from sys import exit
def game():
    # Initialize Pygame modules
    pygame.init()
    pygame.display.set_caption('procognitor')
    # Set Display
    surface = pygame.display.set_mode(size=(1920//2, 1080//2))

    # Set up text for score 
    text = pygame.font.SysFont("Comic Sans MS", 30)

    # Also Works
    # text = pygame.font.Font(None, 30)

    # Hide the mouse so its just a dot
    pygame.mouse.set_visible(False)

    # Initialize variables
    timer = time.time()
    xpos = randrange(50, 700)
    ypos = randrange(50, 450)
    score = 0
    miss = 0
    accuracy = 0


    # Main Body Of Program
    while True:

        #Get keyboard and mouse input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            # Checks to see if target was hit or not
            if event.type == pygame.MOUSEBUTTONDOWN:
                if mouse[0] < xpos + 4 and mouse[0] > xpos - 4 and mouse[1] < ypos + 4 and mouse[1] > ypos - 4:
                    score += 1
                    timer = time.time()
                    xpos = randrange(50, 700)
                    ypos = randrange(50, 450)
                    accuracy = 100 * (score / (score + miss))
                else:
                    miss += 1
                    accuracy = 100 * (score / (score + miss))
        

        mouse = pygame.mouse.get_pos()
        surface.fill((0,0,0))
        pygame.draw.circle(surface, (255,255,255), mouse, 5)

        # Display target
        if time.time() < timer + 2:
            pygame.draw.circle(surface, (255,0,0), (xpos, ypos), 5)
        else: 
            timer = time.time()
            xpos = randrange(50, 700)
            ypos = randrange(50, 450)

        surface.blit(text.render(f"Score: {score}", True, (255,255,255)), (0,0))
        surface.blit(text.render(f"Accuracy: {accuracy}%", True, (255,255,255)), (200,0))

        surface.blit(text.render("This Game is to increase the reaction time of any person which", True, (255,255,255)), (0,450))
        surface.blit(text.render("can help at the time of geomagnetic storm.", True, (255,255,255)), (0,480))

        # Update display
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
                
        pygame.display.update()
