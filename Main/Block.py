from random import randint
import pygame


class Block:

    def __init__(self, lives, x, y, color=None):
        self.lives = lives
        self.x = x
        self.y = y + 62
        self.length = 55
        self.boundries = pygame.Rect(self.x, self.y, self.length, self.length)
        self.color = (255, 0, 0) 

        self.font = pygame.font.SysFont("Arial", 20)

    def decrement_lives(self):
        self.lives -= 1

    def reset_boundaries(self):
        self.boundries = pygame.Rect(self.x, self.y, self.length, self.length)

    def draw_block(self, screen):
        self.reset_boundaries()
        pygame.draw.rect(screen, self.color, self.boundries, 2)
        lives_text = self.font.render(str(self.lives), True, self.color)
        screen.blit(lives_text, (self.x + ((self.length / 2) - 5), self.y + (self.length / 2) - 10))
