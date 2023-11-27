import numpy as np

def step(self, action_angle_index, show=False):
    action_angle = -1 * (((action_angle_index + 1) / 20) * 0.9 * torch.pi + 0.1)  # 切為20等分
    reward = 0
    self.balls_running = True

    # Create an 8x7 empty matrix to represent the game state
    state_matrix = np.zeros((8, 7), dtype=int)

    if show:
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
            if powerup.y >= self.bottom:
                bad_powerups.append(powerup)
                powerup.show = False

            if powerup.show:
                self.screen.blit(powerup.img, (powerup.x, powerup.y))
        self.powerups = [p for p in self.powerups if (p not in bad_powerups)]

        # write level number
        level_text = self.font.render("Level: " + str(self.level), True, (0, 0, 0))
        self.screen.blit(level_text, ((self.screen.get_width() - level_text.get_width() - 20), 455))

        # draw highscore
        high_score = self.font.render("High Score: " + str(self.highscore), True, (0, 0, 0))
        self.screen.blit(high_score, (20, 455))

        # draw baseline
        pygame.draw.line(self.screen, (0, 0, 0), (0, 450), (self.width, 450))

        pygame.display.flip()
    # # print("here")
    ####
    # mouse_pos = pygame.mouse.get_pos()

    # self.ball_angle = math.atan2(mouse_pos[1] - self.balls[0][2], mouse_pos[0] - self.balls[0][1])
    self.ball_angle = action_angle
    # print("self.ball_angle",self.ball_angle)
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
                        break

            # render balls
            balls_above_0 = 0
            for ball in self.balls:
                velx = math.cos(ball[0]) * ball[3]
                vely = math.sin(ball[0]) * ball[4]

                ball[1] += velx
                ball[2] += vely

                if ball[1] > 434:
                    ball[3] = ball[3] * -1
                    ball[1] = 434
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

                ball_rect = self.ball.get_rect()
                ball_rect.x = ball[1]
                ball_rect.y = ball[2]
                bad_blocks = []
                for block in self.blocks:
                    collision = False
                    block_rect = block.boundries
                    block_rect.x = block.x
                    block_rect.y = block.y
                    block_right = block.x + block.length
                    block_left = block.x
                    block_top = block.y
                    block_bottom = block.y + block.length
                    ball_left = ball_rect.x
                    ball_right = ball_rect.x + 20
                    ball_top = ball_rect.y
                    ball_bottom = ball_rect.y + 20

                    if block_left <= ball_right <= block_right and block_bottom >= ball_top >= block_top:
                        collision = True
                    if block_left <= ball_left <= block_right and block_bottom >= ball_top >= block_top:
                        collision = True
                    if block_left <= ball_right <= block_right and block_bottom >= ball_bottom >= block_top:
                        collision = True
                    if block_left <= ball_left <= block_right and block_bottom >= ball_bottom >= block_top:
                        collision = True

                    if ball_rect.colliderect(block_rect) and collision:
                        # 計算碰撞點的位置
                        collision_x = max(block_left, min(ball_rect.centerx, block_right))
                        collision_y = max(block_top, min(ball_rect.centery, block_bottom))

                        # 判斷碰撞發生的方向
                        if ball_rect.centerx < collision_x:
                            ball[1] = block_left - ball_rect.width
                            ball[3] = ball[3] * -1
                        elif ball_rect.centerx > collision_x:
                            ball[1] = block_right
                            ball[3] = ball[3] * -1

                        if ball_rect.centery < collision_y:
                            ball[2] = block_top - ball_rect.height
                            ball[4] = ball[4] * -1
                        elif ball_rect.centery > collision_y:
                            ball[2] = block_bottom
                            ball[4] = ball[4] * -1

                        # 減少磚塊生命值
                        block.decrement_lives()

                        # 檢查磚塊生命值是否歸零
                        if block.lives <= 0:
                            bad_blocks.append(block)

                self.blocks = [b for b in self.blocks if (b not in bad_blocks)]

                for powerup in self.powerups:
                    rect = powerup.img.get_rect()
                    rect.x = powerup.x
                    rect.y = powerup.y
                    if ball_rect.colliderect(rect):
                        powerup_type = powerup.powerup_type
                        if powerup_type == 0:
                            powerup.hit = True
                            powerup.show = False

                # for b in self.balls:
                # self.screen.blit(self.ball, (b[1], b[2]))

                if ball[3] != 0 and ball[4] != 0 and ball[5] == 1:
                    balls_above_0 += 1

            if balls_above_0 == 0 and self.balls_ran == len(self.balls):
                # new level
                self.level += 1
                self.balls_running = False
                self.balls_ran = 0

                # self.ball_velx = self.ball_velx + 0.15
                # self.ball_vely = self.ball_vely + 0.15

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
                reward += self.reward_level_up

        else:
            self.balls_on_ground = False

            # level up reward
            #     reward += self.reward_level_up

            # else:
            #     self.balls_on_ground = False

        # render blocks
        # also get the closest brick position
        max_brick_y = 0
        closest_brick = None
        for block in self.blocks:
            block.draw_block(self.screen)

            if block.y >= self.bottom:
                game = False
            if block.y > max_brick_y:
                closest_brick = block
                max_brick_y = block.y

        if show:
            # display balls
            for ball in self.balls:
                self.screen.blit(self.ball, (ball[1], ball[2]))

            # render power ups
            bad_powerups = []
            for powerup in self.powerups:
                if powerup.y >= self.bottom:
                    bad_powerups.append(powerup)
                    powerup.show = False

                if powerup.show:
                    self.screen.blit(powerup.img, (powerup.x, powerup.y))
            self.powerups = [p for p in self.powerups if (p not in bad_powerups)]

            # write level number
            level_text = self.font.render("Level: " + str(self.level), True, (0, 0, 0))
            self.screen.blit(level_text, ((self.screen.get_width() - level_text.get_width() - 20), 455))

            # draw highscore
            high_score = self.font.render("High Score: " + str(self.highscore), True, (0, 0, 0))
            self.screen.blit(high_score, (20, 455))

            # draw baseline
            pygame.draw.line(self.screen, (0, 0, 0), (0, 450), (self.width, 450))

            pygame.display.flip()

    self.balls_on_ground = False

    if game:
        done = False
    else:
        done = True

        # get reward of die penalty
        # reward += self.reward_game_done_penalty

    # Update the state matrix based on the closest brick position
    if closest_brick is not None:
        # Calculate the row and column of the closest brick in the 8x7 matrix
        row = int(closest_brick.y / 62)
        col = int(closest_brick.x / 62)
        state_matrix[row, col] = 1

    # Convert the state matrix to a 1D array and add the ball position
    state = state_matrix.flatten()
    state = np.insert(state, 0, self.ball_pos[0])

    print(state)

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
