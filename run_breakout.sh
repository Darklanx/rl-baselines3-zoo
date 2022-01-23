for seed in {1..10}
do
    python3 train.py --algo offpac --env breakout -min -n 3000000 --seed $seed -params KL:True 
done