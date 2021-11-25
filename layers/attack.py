from read_yml import parse_layer_from_yml
from utils import load_data
from data import Data
#  parse_layer_from_yml    

if __name__ == '__main__':
    # load data
    data = load_data('data/data/data.pkl')

    # parse layer from yml
    layer = parse_layer_from_yml('configs/attack.yml')
    # attack
    adv = layer.attack(data.X_test)