from BBTan import BBTan


def main():
    game = BBTan()

    angle = -0.8
    for i_episode in range(2):
        while angle > -1.6:

            done, reward, state = game.step(angle)
            angle -= 0.05
            print("angle now",angle,"game_continue",done, "reward", reward, "state", state)

            if done:

                print("game stop, reset / reopen the game")
                game = BBTan()
                break


if __name__ == "__main__":
    main()