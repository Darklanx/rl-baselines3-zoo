# python3 scripts/plot_eval.py -a offpac  --env  MountainCar-v0 -f logs --ids 8 10 11
# python3 scripts/plot_train.py -a offpac --env MountainCar-v0 -f logs --ids 8 10 11
# python3 scripts/plot_eval.py -a offpac  --env  MountainCar-v0 -f logs --ids 12 13 
# python3 scripts/plot_train.py -a offpac --env MountainCar-v0 -f logs --ids 12 13
# python3 scripts/plot_eval.py -a offpac  --env  MountainCar-v0 -f logs --ids 16
# python3 scripts/plot_train.py -a offpac --env MountainCar-v0 -f logs --ids 16
python3 scripts/plot_eval.py -a offpac  --env  BreakoutNoFrameskip-v4 -f logs  --ids 1 3 4 
python3 scripts/plot_train.py -a offpac --env BreakoutNoFrameskip-v4 -f logs  --ids 1 3 4

# python3 scripts/plot_train.py -a offpac --env CartPole-v0 -f logs  
