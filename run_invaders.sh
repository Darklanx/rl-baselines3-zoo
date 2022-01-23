for seed in {1..10}
do
    python3 train.py --algo offpac --env space_invaders -min -n 3000000 --seed $seed -params KL:True 
done