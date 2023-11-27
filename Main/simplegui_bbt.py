import SimpleGUICS2Pygame.simpleguics2pygame as simplegui
import math
import random
import pyautogui

WIDTH = 350  # 視窗寬度
HEIGHT = 600  # 視窗高度
BALL_RADIUS = 10  # 球的半徑
BALL_VEL = 10   # 球的速度
CORNER_DIR = [[1, 1], [1, -1], [-1, -1], [-1, 1]]  # 方塊的四個角的方向
TOP_HEIGHT = 45  # 頂部的高度
BOTTOM_HEIGHT = 505  # 底部的高度
TILE_RADIUS = 20  # 方塊的半徑
BONUS_TILE_RADIUS = 10  # 獎勵方塊的半徑
time = 0  # 時間
started = False  # 遊戲是否開始
ball_number = 1  # 球的數量
level = 0  # 等級
tile_set = set()  # 方塊集合
ball_set = set()  # 球集合
position = []  # 位置
angle = math.pi * 1.5  # 角度
del_angle = 0  # 角度變化
balls_to_shoot = 0  # 需要射擊的球數量
arrow_len = 100  # 箭頭長度
del_arrow_len = 0  # 箭頭長度變化
ready_next = False  # 是否準備好下一個球
next_pos0 = 0  # 下一個球的位置
first_die = False  # 是否第一次死亡
colliding = 0  # 碰撞計數 

def angle_to_dir(ang):
    # 將角度轉換為方向向量
    return [math.cos(ang), math.sin(ang)]

class Ball:
    def __init__(self, pos, dir):
        self.pos = list(pos)  # 球的位置
        self.dir = list(dir)  # 球的方向
        
    def draw(self, canvas):
        # 繪製球
        canvas.draw_circle(self.pos, BALL_RADIUS, 1, 'White', 'White')
        
    def update(self):
        # 更新球的位置
        new_pos = [self.pos[0] + self.dir[0] * BALL_VEL, 
                   self.pos[1] + self.dir[1] * BALL_VEL]
        # 檢查是否撞到邊緣
        if new_pos[0] >= WIDTH or new_pos[0] <= 0:
            self.dir[0] = -self.dir[0]
        if new_pos[1] >= HEIGHT or new_pos[1] <= TOP_HEIGHT:
            self.dir[1] = -self.dir[1]
        new_pos = [self.pos[0] + self.dir[0] * BALL_VEL, 
                   self.pos[1] + self.dir[1] * BALL_VEL]
        self.pos = new_pos
        
    def die(self):
        global first_die, next_pos0

        # 檢查是否死亡
        if self.pos[1] > BOTTOM_HEIGHT:
            if first_die:
                first_die = False
                next_pos0 = self.pos[0]
            return True
        else:
            return False
             
    def collide(self, tile):
        global colliding  
        # 獲取方塊的位置
        tile_pos = tile.get_pos()
        del_x = abs(self.pos[0] - tile_pos[0])
        del_y = abs(self.pos[1] - tile_pos[1])
        x_in = del_x <= TILE_RADIUS + BALL_RADIUS
        y_in = del_y <= TILE_RADIUS + BALL_RADIUS
        if x_in and y_in:
            # 檢查球的邊緣是否與方塊的邊緣相交
            if (del_x > TILE_RADIUS) and (del_x <= TILE_RADIUS + BALL_RADIUS):
                # [x軸(0)]相交時，去修正球的位置
                if self.pos[0] < tile_pos[0]:
                    self.pos[0] = tile_pos[0] - (TILE_RADIUS + BALL_RADIUS)
                else:
                    self.pos[0] = tile_pos[0] + (TILE_RADIUS + BALL_RADIUS)

            if (del_y > TILE_RADIUS) and (del_y <= TILE_RADIUS + BALL_RADIUS):
                # [y軸(1)] 相交時，去修正球的位置
                if self.pos[1] < tile_pos[1]:
                    self.pos[1] = tile_pos[1] - (TILE_RADIUS + BALL_RADIUS)
                else:
                    self.pos[1] = tile_pos[1] + (TILE_RADIUS + BALL_RADIUS)
            
            if not tile.transparent: # 判斷是碰到Tile 還是 Bouns Tile
                if del_x >= del_y:
                    colliding += 1
                    self.dir[0] = -self.dir[0]

                if del_x <= del_y:
                    colliding -= 1
                    self.dir[1] = -self.dir[1]
            
            return True
        else:
            return False

class Tile:
    def __init__(self, pos, start_level):
        self.pos = list(pos)  # 方塊的位置
        self.life = start_level  # 方塊的生命值
        self.start_level = start_level  # 方塊的初始等級
        self.transparent = False  # 方塊是否透明
        self.multiplier_applied = False  # 乘法標記判斷
        global game_over_printed
        game_over_printed = False
        
    def get_pos(self):
        global level

        # 獲取方塊的位置
        self.pos[1] = level - self.start_level + 2  # 方塊向下移動一行
        return [25 + self.pos[0] * 50, 25 + self.pos[1] * 50]
    
    def draw(self, canvas):        
        # 繪製方塊
        center_pos = self.get_pos()
        point_list = []
        for i in range(4):
            point_list.append([center_pos[0] + CORNER_DIR[i][0] * TILE_RADIUS, 
                               center_pos[1] + CORNER_DIR[i][1] * TILE_RADIUS])
        canvas.draw_polygon(point_list, 3, 'Red')
        
        text_pos = [center_pos[0] - 5, center_pos[1] + 6]
        if self.life >= 10:
            text_pos[0] -= 6
            if self.life >= 100:
                text_pos[0] -= 6
        canvas.draw_text(str(self.life), text_pos, 20, 'Red')
    
    def update(self):
        global started, game_over_printed, level

        self.get_pos()
        if self.pos[1] >= 9 and not game_over_printed:
            print('Level: ' + str(level - 1))  # 死之前到達的等級
            game_over_printed = True
            started = False
            frame.stop()
            timer.stop()
        
        # 判斷是否到達10的倍數，並且只有在當前等級方塊生成時才應用乘法運算
        if level % 10 == 0 and self.start_level == level and not self.multiplier_applied:
            self.life *= 2
            self.multiplier_applied = True  # 乘法計算 = Ture
    
    def die(self):
        # 判斷方塊是否死亡
        return self.life <= 0
    def got_hit(self):
        self.life -= 1

class Bonus(Tile): # 獎勵方塊
    def __init__(self, start_level, x): # 初始化函數，接受 start_level 和 x座標作為參數
        super().__init__([x, 0], start_level) # 調用父類 Tile 的初始化函數，設定方塊的位置和起始等級
        self.color = 'Yellow'
        self.life = 1
        self.transparent = True
        self.radius = 10  # 設定半徑大小

    def draw(self, canvas):
        center_pos = self.get_pos()

        # 繪製圓形
        canvas.draw_circle(center_pos, self.radius, 3, self.color)

        text_pos = [center_pos[0] - 5, center_pos[1] + 6]
    
    def die(self):
        global ball_number
        if self.life <= 0:
            ball_number += 1
        return super().die()

def fate():
    return random.randrange(10) < 5 #random (1~10) (50%)

def next_level():
    global level    
    level += 1
    
    # tile gerenate
    for i in range(6):
        if fate():
            tile_set.add(Tile([i, 0], level))

    
    # Bonus_tile gerenate
    bouns_tile = random.randint(0, 6)  # 隨機選擇bonus_tile的 x 座標
    while [bouns_tile, 0] in [[tile.pos[0], tile.pos[1]] for tile in tile_set]: 
        bouns_tile = random.randint(0, 6)
    tile_set.add(Bonus(level, bouns_tile))
    
def process_group(group, canvas):
    for item in set(group):
        item.draw(canvas)
        item.update()
        if item.die():
            group.remove(item)
        
def draw_arrow(canvas):
    global angle, position, arrow_len, del_arrow_len
    
    dir = angle_to_dir(angle)
    new_arrow_len = arrow_len + del_arrow_len
    if 25 <= new_arrow_len <= BOTTOM_HEIGHT - TOP_HEIGHT:
        arrow_len = new_arrow_len
    tip_pos = [position[0] + dir[0] * arrow_len, position[1] + dir[1] * arrow_len]
    dir_left, dir_right = angle_to_dir(angle - math.pi * 0.75), angle_to_dir(angle + math.pi * 0.75)
    lr_len = 7
    left_pos =  [tip_pos[0] +  dir_left[0] * lr_len, tip_pos[1] +  dir_left[1] * lr_len]
    right_pos = [tip_pos[0] + dir_right[0] * lr_len, tip_pos[1] + dir_right[1] * lr_len]
    
    canvas.draw_line(position, tip_pos, 2, 'White')
    canvas.draw_line(tip_pos, left_pos, 2, 'White')
    canvas.draw_line(tip_pos, right_pos, 2, 'White')
    
def ball_hit_group(ball, group):
    for item in set(group):
        if ball.collide(item):
            item.got_hit()
            if item.die():
                group.remove(item)
    
def draw(canvas):
    global started, time, tile_set, ball_set, angle, del_angle, level, colliding
    
    if not started:
        canvas.draw_text('BB-TAN', [50, 300], 70, 'Yellow')
        canvas.draw_text('Press SPACE to play', [90, 500], 20, 'White')
    else:
        # 更新
        process_group(tile_set, canvas)
        process_group(ball_set, canvas)
        if math.pi * 1.1 < angle + del_angle < math.pi * 1.9:
            angle = (angle + del_angle) % (math.pi * 2)
        
        # 上下界線
        canvas.draw_line([0, TOP_HEIGHT], [WIDTH, TOP_HEIGHT], 3, 'White')
        canvas.draw_line([0, BOTTOM_HEIGHT], [WIDTH, BOTTOM_HEIGHT], 3, 'White')
        
        # 箭頭
        draw_arrow(canvas)
        
        # 等級/目前球的數量
        canvas.draw_text("Level " + str(level), [WIDTH / 2 - 35, TOP_HEIGHT - 15], 20, 'White')
        canvas.draw_text(str(ball_number) + ' balls', [WIDTH / 2 - 35, BOTTOM_HEIGHT + 35], 20, 'White')
        
        # 碰撞
        for ball in ball_set:
            ball_hit_group(ball, tile_set)
            
def new_game():
    global started, ball_number, level, tile_set, ball_set, position, angle, del_angle, balls_to_shoot
    
    started = True
    ball_number = 1
    level = 0
    tile_set = set()
    ball_set = set()
    position = [WIDTH / 2, BOTTOM_HEIGHT]
    angle = math.pi * 1.5
    del_angle = 0
    balls_to_shoot = 0
    ready_next = False
    
    next_level()

    
    
def shoot():
    global ball_number, balls_to_shoot, first_die
    
    # ball_number += 1 # 每回合增加球數量
    first_die = True
    balls_to_shoot = ball_number
    
def keydown(key):
    global started, del_angle

    # shooting, cannot move
    if len(ball_set) > 0:
        return
    
    if key == simplegui.KEY_MAP['space']:
        if not started:
            new_game()
        else:
            shoot()
    elif key == simplegui.KEY_MAP['left']:
        del_angle = -0.02
    elif key == simplegui.KEY_MAP['right']:
        del_angle = 0.02

    
def keyup(key):
    global del_angle
    
    if key == simplegui.KEY_MAP['left']:
        del_angle = 0
    elif key == simplegui.KEY_MAP['right']:
        del_angle = 0

def timer_handler():
    global time, position, angle, balls_to_shoot, ball_set, ready_next, next_pos0
    
    time += 1
    if time % 20 == 0 and balls_to_shoot > 0:
        ball_set.add( Ball(position, angle_to_dir(angle)) )
        balls_to_shoot -= 1
        if balls_to_shoot == 0:
            ready_next = True
            

    if ready_next and len(ball_set) == 0:
        ready_next = False
        position[0] = next_pos0
        next_level()


frame = simplegui.create_frame('BBTAN', WIDTH, HEIGHT)
frame.set_canvas_background('Black')
frame.set_draw_handler(draw)
frame.set_keydown_handler(keydown)
frame.set_keyup_handler(keyup)

timer = simplegui.create_timer(1, timer_handler)

timer.start()
frame.start() 