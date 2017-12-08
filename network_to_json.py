import argparse
import pickle

def args():
    parser = argparse.ArgumentParser(description='Convert network to json.')
    parser.add_argument('network_path', help='Path for network pickle.')
    return parser.parse_args()

if __name__ == "__main__":
    args = args()
    network = pickle.load(open(args.network_path, 'rb'))
    print(network.to_json())
