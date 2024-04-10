import torch
from Game.PlayerSection.basic import main_basic
from Game.PlayerSection.player_mc import main_mc
from Game.PlayerSection.player_mcmc import main_mcmc
from Game.player import main_game


user_choice = input("Do you want to play the game or have AI? (Enter 'play' or 'AI'): ").lower()

if user_choice == "play":
    main_game()
elif user_choice == "ai":
    user_choice = input("Do you want to Train or not? Yes/No:  ").lower()
    if user_choice == 'yes':
        main_basic(True, 50)
        main_mc(True, 50)
        main_mcmc(True, 50)
    else:
        print("Basic model running")
        main_basic(False, 0)

        print("MC model running")
        main_mc(False, 0)

        print("MCMC model runnig")
        main_mcmc(False, 0)
else:
    print("Invalid choice. Please enter 'play' or 'AI'.")