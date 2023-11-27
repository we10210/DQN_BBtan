import pygame


class PowerUp:

    def __init__(self, screen, powerup_type, x, y):
        self.screen = screen
        self.powerup_type = powerup_type
        self.x = x
        self.y = y + 62
        self.hit = False
        self.show = True

        if self.powerup_type == 0:
            self.img_path = 'imgs/add.png'
        else:
            self.img_path = 'imgs/explosion.png'

        self.img = pygame.image.load(self.img_path)
        self.img = pygame.transform.scale(self.img, (20, 20))