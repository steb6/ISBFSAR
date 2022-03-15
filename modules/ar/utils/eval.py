import pickle
from modules.ar.trx import ActionRecognizer
from utils.matplotlib_visualizer import MPLPosePrinter
from utils.params import TRXConfig

if __name__ == "__main__":  # Test accuracy

    with open('assets/skeleton_types.pkl', "rb") as input_file:
        skeleton_types = pickle.load(input_file)
    skeleton = 'smpl+head_30'
    edges = skeleton_types[skeleton]['edges']

    # NORMAL
    ar = ActionRecognizer(TRXConfig())
    vis = MPLPosePrinter()
    path = TRXConfig().data_path

    query = 'clapping'
    support = ['clapping', 'drop', 'throw', 'pickup', 'hand_waving']

    with open(path+query+"/10.pkl", "rb") as infile:
        q = pickle.load(infile)

    with open(path+support[0]+"/8.pkl", "rb") as infile:
        s1 = pickle.load(infile)
        ar.train((s1, support[0]))

    with open(path+support[1]+"/8.pkl", "rb") as infile:
        s2 = pickle.load(infile)
        ar.train((s2, support[1]))

    with open(path+support[2]+"/8.pkl", "rb") as infile:
        s3 = pickle.load(infile)
        ar.train((s3, support[2]))

    with open(path+support[3]+"/8.pkl", "rb") as infile:
        s4 = pickle.load(infile)
        ar.train((s4, support[3]))

    with open(path+support[4]+"/8.pkl", "rb") as infile:
        s5 = pickle.load(infile)
        ar.train((s5, support[4]))

    vis.set_title("QUERY:"+query)
    for elem in q:
        vis.clear()
        vis.print_pose(elem, edges)
        vis.sleep(0.1)

    for seq, c in zip([s1, s2, s3, s4, s5], support):
        vis.set_title("SUPPORT:"+c)
        for elem in seq:
            vis.clear()
            vis.print_pose(elem, edges)
            vis.sleep(0.1)

    for elem in q:
        print(ar.inference(elem))
