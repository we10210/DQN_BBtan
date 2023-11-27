import math
import pygame
from random import randint
from Block import Block
from PowerUp import PowerUp
from Stats import Stats
import numpy as np
import torch

class BBTan:

    def __init__(self):
        
        # load game properties
        self.level = 1
        self.current_level = 1
        self.max_blocks = 2
        self.balls_running = False
        self.balls_on_ground = False
        self.bottom = 530
        self.ball_velx = 20 # 速度
        self.ball_vely = 20 # 速度
        self.blocks = []
        self.balls = []
        self.width = 432
        self.height = 600
        self.ball_angle = 0
        self.balls_ran = 0
        self.ball_timer = 0
        self.ball_delay = 150
        self.stats = Stats()
        self.highscore = self.stats.read_score()
        self.powerups = []
        self.state_matrix = np.zeros((8, 7), dtype=int) #建立8*7的矩陣
        self.ball_pos = [217, self.bottom]
        
        # 正reward 
        # self.reward_collision_brick = 5        #brick碰撞
        # self.reward_collision_bouns_ball = 50   #bouns ball
        self.reward_level_up = 1                #每升一級
        # self.reward_to_level_8 = 300            #超過level 8                    

        # 負reward
        # self.reward_game_done_penalty = -500    #遊戲結束


        # init pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.SysFont("Arial", 20)

        # render first block reshape
        self.possible_block_positions = [0, 62, 124, 186, 248, 310, 372]

        # get a random number of blocks
        self.add_new_row() 

        # load imgs 
        self.ball = pygame.image.load("imgs/soccer_ball.png")
        self.ball = pygame.transform.scale(self.ball, (20, 20))

        self.gameover = pygame.image.load("imgs/gameover.png")

        # init starting balls
        self.balls.append([0, self.ball_pos[0], self.ball_pos[1], 0, 0, 0])
        self.screen.blit(self.ball, (self.balls[0][1], self.balls[0][2]))

        pygame.display.flip()

    def get_init_state(self): # 初始的State資訊在此

        init_matrix = np.zeros((8, 7), dtype=int).reshape(-1)
        init_ball_position = np.array([self.ball_pos[0]])
        init_state = np.concatenate((init_matrix, init_ball_position), axis=0)

        return init_state

    def add_new_row(self):
        total_num_blocks = randint(2, 5)
        taken_blocks = []
        num_blocks = 0

        pos = randint(0, len(self.possible_block_positions) - 1)
        taken_blocks.append(pos)
        new_powerup = PowerUp(self.screen, 0, self.possible_block_positions[pos] + 20, 25)
        self.powerups.append(new_powerup)

        while num_blocks < total_num_blocks:
            pos = randint(0, len(self.possible_block_positions) - 1)
            if not (pos in taken_blocks):
                block = Block(self.level * 1 if self.level % 10 == 0 else self.level,
                              self.possible_block_positions[pos], 0, color=(255, 0, 0) if self.level % 10 == 0 else None)
                self.blocks.append(block)
                taken_blocks.append(pos)
                num_blocks += 1
    
    def record_game_state(self): # 紀錄矩陣
        # 清空
        self.state_matrix.fill(0)

        # BRICK = 1
        for block in self.blocks:
            row = int(block.y // 62)  # brick的行
            col = int(block.x // 62)  # brick的列
            if 0 <= row < 8 and 0 <= col < 7:
                self.state_matrix[row, col] = block.lives

        # powerup = -1
        for powerup in self.powerups:
            if powerup.show:
                row = int(powerup.y // 62)  # powerup的行
                col = int(powerup.x // 62)  # powerup的列
                if 0 <= row < 8 and 0 <= col < 7:
                    self.state_matrix[row, col] = -1
    
    def close_game(self):
        pygame.quit()
    
    def reset_game(self):
        self.level = 1
        self.current_level = 1
        self.max_blocks = 2
        self.balls_running = False
        self.balls_on_ground = False
        self.blocks = []
        self.balls = []
        self.ball_angle = 0
        self.balls_ran = 0
        self.ball_timer = 0
        self.ball_delay = 150
        self.powerups = []
        self.ball_pos = [217, self.bottom]
        
        # 正reward 
        # self.reward_collision_brick = 5        #brick碰撞
        # self.reward_collision_bouns_ball = 50   #bouns ball
        self.reward_level_up = 1                #每升一級
        # self.reward_to_level_8 = 300            #超過level 8 
        # render first block
        self.possible_block_positions = [0, 62, 124, 186, 248, 310, 372]

        # get a random number of blocks
        self.add_new_row() 

        # init starting balls
        self.balls.append([0, self.ball_pos[0], self.ball_pos[1], 0, 0, 0])
        self.screen.blit(self.ball, (self.balls[0][1], self.balls[0][2]))

        pygame.display.flip()

                
        
    def step(self, action_angle_index, show = True): #DQN Train
        action_angle = -1 * ( ((action_angle_index+1) / 20) * 0.9 * torch.pi +0.15) # action能做的選擇 有20個角度可以選(0~19)
        # +print('action_angle',action_angle)
        reward = 0
        self.balls_running = True
        self.record_game_state()
        # print_state = True
        

        if show:
            if self.balls_ran == 0:
                self.record_game_state
            ####
            self.screen.fill((255, 255, 255))
            # render blocks
            for block in self.blocks:
                block.draw_block(self.screen)

            # display balls
            for ball in self.balls:
                self.screen.blit(self.ball, (ball[1], ball[2]))

            # render power ups
            bad_powerups = []
            for powerup in self.powerups:
                if powerup.y >= 510:
                    bad_powerups.append(powerup)
                    powerup.show = False

                if powerup.show:
                    self.screen.blit(powerup.img, (powerup.x, powerup.y))
            self.powerups = [p for p in self.powerups if (p not in bad_powerups)]

            # write level number
            level_text = self.font.render("Level: " + str(self.level), True, (0, 0, 0))
            self.screen.blit(level_text, ((self.screen.get_width() - level_text.get_width() - 20), 555))

            # draw highscore
            high_score = self.font.render("High Score: " + str(self.highscore), True, (0, 0, 0))
            self.screen.blit(high_score, (20, 555))

            # draw baseline
            pygame.draw.line(self.screen, (0, 0, 0), (0, 550), (self.width, 550))

            pygame.display.flip()
        # # print("here")
        ####
        # mouse_pos = pygame.mouse.get_pos()

        # self.ball_angle = math.atan2(mouse_pos[1] - self.balls[0][2], mouse_pos[0] - self.balls[0][1])
        self.ball_angle = action_angle
        self.balls[0][0] = self.ball_angle

        # calc new velocities
        if self.ball_angle < math.pi / 2:
            self.balls[0][3] = self.ball_velx
        else:
            self.balls[0][3] = self.ball_velx * -1

        self.balls[0][4] = self.ball_vely
        self.balls[0][5] = 1
        self.ball_timer = pygame.time.get_ticks()
        self.balls_ran += 1

        game = True

        # while game:

        # update the screen util ball is not running
        while self.balls_running:
            self.screen.fill((255, 255, 255))

            if self.balls_running:
                # balls are hitting blocks

                if pygame.time.get_ticks() - self.ball_timer > self.ball_delay and self.balls_ran < len(self.balls):
                    for ball in self.balls:
                        if ball[3] == 0 and ball[4] == 0 and ball[5] == 0:
                            ball[0] = self.ball_angle

                            # calc new velocities
                            if self.ball_angle < math.pi / 2:
                                ball[3] = self.ball_velx
                            else:
                                ball[3] = self.ball_velx * -1

                            ball[4] = self.ball_vely
                            ball[5] = 1
                            self.ball_timer = pygame.time.get_ticks()
                            self.balls_ran += 1
                            print_state = True
                            break
                                   
                # render balls 
                balls_above_0 = 0
                for ball in self.balls:
                    velx = math.cos(ball[0]) * ball[3]
                    vely = math.sin(ball[0]) * ball[4]

                    ball[1] += velx
                    ball[2] += vely

                    if ball[1] > 432:
                        ball[3] = ball[3] * -1
                        ball[1] = 432
                    elif ball[1] < 0:
                        ball[3] = ball[3] * -1
                        ball[1] = 0
                    elif ball[2] < 0:
                        ball[4] = ball[4] * -1
                        ball[2] = 0
                    elif ball[2] > self.bottom:
                        if not self.balls_on_ground:
                            self.balls_on_ground = True
                            self.ball_pos[0] = ball[1]

                        ball[3] = 0
                        ball[4] = 0
                        ball[2] = self.bottom
                        ball[1] = self.ball_pos[0]

                    ball_rect = self.ball.get_rect(topleft=(ball[1], ball[2]))
                    bad_blocks = []
                    for block in self.blocks:
                        block_rect = block.boundries
                        block_rect.x = block.x
                        block_rect.y = block.y

                        if ball_rect.colliderect(block_rect):
                            # Calculate overlap on x and y axis
                            overlap_x = min(ball_rect.right, block_rect.right) - max(ball_rect.left, block_rect.left)
                            overlap_y = min(ball_rect.bottom, block_rect.bottom) - max(ball_rect.top, block_rect.top)

                            # Determine collision direction based on the overlap
                            if overlap_x < overlap_y:
                                # Collision happened on the X axis
                                # Determine from which side the collision happened
                                if ball_rect.centerx < block_rect.centerx:
                                    # Ball came from left
                                    ball[1] -= overlap_x
                                else:
                                    # Ball came from right
                                    ball[1] += overlap_x
                                ball[3] = ball[3] * -1
                            else:
                                # Collision happened on the Y axis
                                if ball_rect.centery < block_rect.centery:
                                    # Ball came from top
                                    ball[2] -= overlap_y
                                else:
                                    # Ball came from bottom
                                    ball[2] += overlap_y
                                ball[4] = ball[4] * -1

                            # Decrement block's life
                            block.decrement_lives()
                            
                            # Check if block should be removed
                            if block.lives <= 0:
                                self.blocks.remove(block)
                            break
                        
                                
                    self.blocks = [b for b in self.blocks if b not in bad_blocks]
                    for powerup in self.powerups:
                        rect = powerup.img.get_rect()
                        rect.x = powerup.x
                        rect.y = powerup.y
                        if ball_rect.colliderect(rect):
                            powerup_type = powerup.powerup_type
                            if powerup_type == 0:
                                powerup.hit = True
                                powerup.show = False

                    #for b in self.balls:
                        #self.screen.blit(self.ball, (b[1], b[2]))

                    if ball[3] != 0 and ball[4] != 0 and ball[5] == 1:
                        balls_above_0 += 1

                if balls_above_0 == 0 and self.balls_ran == len(self.balls):
                    # new level
                    self.level += 1
                    self.balls_running = False
                    self.balls_ran = 0

                    #self.ball_velx = self.ball_velx + 0.15
                    #self.ball_vely = self.ball_vely + 0.15

                    for ball in self.balls:
                        ball[5] = 0

                    # move current blocks down
                    for block in self.blocks:
                        block.y = block.y + 62

                    bad_powerups = []
                    for powerup in self.powerups:
                        powerup.y = powerup.y + 62
                        if powerup.hit:
                            if powerup.powerup_type == 0:
                                self.balls.append([0, self.ball_pos[0], self.ball_pos[1], 0, 0, 0])
                            bad_powerups.append(powerup)

                    self.powerups = [p for p in self.powerups if (p not in bad_powerups)]
                    self.add_new_row()
                    reward += self.reward_level_up # 每升一個level 獲得一個+1的reward

            else:
                self.record_game_state()
                state = [self.state_matrix, self.ball_pos[0]]
                # if print_state:
                #     state = [self.state_matrix, self.ball_pos[0]]
                #     print(state, "\n")  # 在第一個level print一次state
                #     print_state = False

                self.balls_on_ground = False

                # level up reward
                #     reward += self.reward_level_up

                # else:
                #     self.balls_on_ground = False

            # render blocks
            # also get the closest brick position
            max_brick_y = 0
            for block in self.blocks:
                block.draw_block(self.screen)

                if block.y >= 450:
                    game = False
                    reward = 0 # 遊戲結束獲得+0的reward

                if block.y > max_brick_y :
                    max_brick_y = block.y

            if show:
            # display balls
                for ball in self.balls:
                    self.screen.blit(self.ball, (ball[1], ball[2]))

                bad_powerups = []
                for powerup in self.powerups:
                    if powerup.y >= 510:
                        bad_powerups.append(powerup)
                        powerup.show = False

                    if powerup.show:
                        self.screen.blit(powerup.img, (powerup.x, powerup.y))
                self.powerups = [p for p in self.powerups if (p not in bad_powerups)]

                # write level number
                level_text = self.font.render("Level: " + str(self.level), True, (0, 0, 0))
                self.screen.blit(level_text, ((self.screen.get_width() - level_text.get_width() - 20), 555))

                # draw bbtan
                # bbtan_text = self.font.render("BBTAN", True, (0, 0, 0))
                # self.screen.blit(bbtan_text, ((self.screen.get_width() / 2 - level_text.get_width() - 20), 555))

                # draw highscore
                high_score = self.font.render("High Score: " + str(self.highscore), True, (0, 0, 0))
                self.screen.blit(high_score, (20, 555))

                # draw baseline
                pygame.draw.line(self.screen, (0, 0, 0), (0, 550), (self.width, 550))

                pygame.display.flip()
        
        
        self.balls_on_ground = False

        if game:
            done = False 
        else:
            done = True
            
            # get reward of die penalty
            # reward += self.reward_game_done_penalty

        # 在這輸入state

        state = np.concatenate((self.state_matrix.reshape(-1), np.array([self.ball_pos[0]])), axis=0)

        # record stats
        if self.level > self.highscore:
            self.stats.record_score(self.level)
        self.stats.calc_avg(self.level)

        if self.level > self.current_level:
            self.current_level = self.level
        # if self.current_level == 8:
            # react level 8
            # reward += self.reward_to_level_8

        return done, reward, state


    def play_game(self): #GUI
        game = True
        print_state = True #state可視化

        while game:
            
            self.screen.fill((255, 255, 255))
            

            if self.balls_running:
                
                # balls are hitting blocks 
                self.record_game_state()
                if pygame.time.get_ticks() - self.ball_timer > self.ball_delay and self.balls_ran < len(self.balls):
                    
                     

                    for ball in self.balls:
                        if ball[3] == 0 and ball[4] == 0 and ball[5] == 0:
                            ball[0] = self.ball_angle

                            # calc new velocities
                            if self.ball_angle < math.pi / 2:
                                ball[3] = self.ball_velx
                            else:
                                ball[3] = self.ball_velx * -1

                            ball[4] = self.ball_vely
                            ball[5] = 1
                            self.ball_timer = pygame.time.get_ticks()
                            self.balls_ran += 1
                            break

                # render balls 
                balls_above_0 = 0
                for ball in self.balls:
                    velx = math.cos(ball[0]) * ball[3]
                    vely = math.sin(ball[0]) * ball[4]

                    ball[1] += velx
                    ball[2] += vely

                    if ball[1] > 420:
                        ball[3] = ball[3] * -1
                        ball[1] = 420
                    elif ball[1] < 0:
                        ball[3] = ball[3] * -1
                        ball[1] = 0
                    elif ball[2] < 0:
                        ball[4] = ball[4] * -1
                        ball[2] = 0
                    elif ball[2] > self.bottom:
                        if not self.balls_on_ground:
                            self.balls_on_ground = True
                            self.ball_pos[0] = ball[1]

                        ball[3] = 0
                        ball[4] = 0
                        ball[2] = self.bottom
                        ball[1] = self.ball_pos[0]

                    ball_rect = self.ball.get_rect(topleft=(ball[1], ball[2]))
                    bad_blocks = []
                    for block in self.blocks:
                        block_rect = block.boundries
                        block_rect.x = block.x
                        block_rect.y = block.y

                        if ball_rect.colliderect(block_rect):
                            # Calculate overlap on x and y axis
                            overlap_x = min(ball_rect.right, block_rect.right) - max(ball_rect.left, block_rect.left)
                            overlap_y = min(ball_rect.bottom, block_rect.bottom) - max(ball_rect.top, block_rect.top)

                            # Determine collision direction based on the overlap
                            if overlap_x < overlap_y:
                                # Collision happened on the X axis
                                # Determine from which side the collision happened
                                if ball_rect.centerx < block_rect.centerx:
                                    # Ball came from left
                                    ball[1] -= overlap_x
                                else:
                                    # Ball came from right
                                    ball[1] += overlap_x
                                ball[3] = ball[3] * -1
                            else:
                                # Collision happened on the Y axis
                                if ball_rect.centery < block_rect.centery:
                                    # Ball came from top
                                    ball[2] -= overlap_y
                                else:
                                    # Ball came from bottom
                                    ball[2] += overlap_y
                                ball[4] = ball[4] * -1

                            # Decrement block's life
                            block.decrement_lives()
                            
                            # Check if block should be removed
                            if block.lives <= 0:
                                self.blocks.remove(block)
                            break
                        
                                
                    self.blocks = [b for b in self.blocks if b not in bad_blocks]
                    for powerup in self.powerups:
                        rect = powerup.img.get_rect()
                        rect.x = powerup.x
                        rect.y = powerup.y
                        if ball_rect.colliderect(rect):
                            powerup_type = powerup.powerup_type
                            if powerup_type == 0:
                                powerup.hit = True
                                powerup.show = False

                    #for b in self.balls:
                        #self.screen.blit(self.ball, (b[1], b[2]))

                    if ball[3] != 0 and ball[4] != 0 and ball[5] == 1:
                        balls_above_0 += 1
                


                if balls_above_0 == 0 and self.balls_ran == len(self.balls):
                    # new level
                    self.level += 1
                    self.balls_running = False
                    self.balls_ran = 0

                    #self.ball_velx = self.ball_velx + 0.15
                    #self.ball_vely = self.ball_vely + 0.15

                    for ball in self.balls:
                        ball[5] = 0

                    # move current blocks down
                    for block in self.blocks:
                        block.y = block.y + 62

                    bad_powerups = []
                    for powerup in self.powerups:
                        powerup.y = powerup.y + 62
                        if powerup.hit:
                            if powerup.powerup_type == 0:
                                self.balls.append([0, self.ball_pos[0], self.ball_pos[1], 0, 0, 0])
                            bad_powerups.append(powerup)

                    self.powerups = [p for p in self.powerups if (p not in bad_powerups)]
                    self.add_new_row()
                    # print(state)

            else:
                self.record_game_state()

                if print_state:
                    state = [self.state_matrix, self.ball_pos[0]]
                    print(state, "\n")  # 新回合開始時print
                    print_state = False

                self.balls_on_ground = False
            
            # render blocks
            for block in self.blocks:
                block.draw_block(self.screen)

                if block.y >= 450:
                    game = False

            # display balls
            for ball in self.balls:
                self.screen.blit(self.ball, (ball[1], ball[2]))

            # render power ups
            bad_powerups = []
            for powerup in self.powerups:
                if powerup.y >= 510:
                    bad_powerups.append(powerup)
                    powerup.show = False

                if powerup.show:
                    self.screen.blit(powerup.img, (powerup.x, powerup.y))
            self.powerups = [p for p in self.powerups if (p not in bad_powerups)]

            # 這邊輸入state
            # state = [self.state_matrix, self.ball_pos[0]]

            # state to NumPy
            # state = np.array(state)
            # print(state)


            # write level number
            level_text = self.font.render("Level: " + str(self.level), True, (0, 0, 0))
            self.screen.blit(level_text, ((self.screen.get_width() - level_text.get_width() - 20), 555))

            # draw bbtan
            # bbtan_text = self.font.render("BBTAN", True, (0, 0, 0))
            # self.screen.blit(bbtan_text, ((self.screen.get_width() / 2 - level_text.get_width() - 20), 555))

            # draw highscore
            high_score = self.font.render("High Score: " + str(self.highscore), True, (0, 0, 0))
            self.screen.blit(high_score, (20, 555))

            # draw baseline
            pygame.draw.line(self.screen, (0, 0, 0), (0, 550), (self.width, 550))

            pygame.display.flip()

            # user selecting where to shoot
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit(0)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.balls_running:
                        self.balls_running = True
                        mouse_pos = pygame.mouse.get_pos()

                        self.ball_angle = math.atan2(mouse_pos[1] - self.balls[0][2], mouse_pos[0] - self.balls[0][1])
                        self.balls[0][0] = self.ball_angle

                        # calc new velocities
                        if self.ball_angle < math.pi / 2:
                            self.balls[0][3] = self.ball_velx
                        else:
                            self.balls[0][3] = self.ball_velx * -1

                        self.balls[0][4] = self.ball_vely
                        self.balls[0][5] = 1
                        self.ball_timer = pygame.time.get_ticks()
                        self.balls_ran += 1
                        print_state = True #開啟

        pygame.font.init()
        font = pygame.font.Font(None, 24)
        text = font.render("Score: " + str(self.level), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.centerx = self.screen.get_rect().centerx
        textRect.centery = self.screen.get_rect().centery + 24
        self.screen.blit(self.gameover, (0, 0))
        self.screen.blit(text, textRect)
        

        # record stats
        if self.level > self.highscore:
            self.stats.record_score(self.level)
        self.stats.calc_avg(self.level)

        self.ball_timer = pygame.time.get_ticks()
        while pygame.time.get_ticks() - self.ball_timer < 5000:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit(0)
            pygame.display.flip()
