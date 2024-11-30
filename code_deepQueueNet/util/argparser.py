import argparse



def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=int,default=1,help='1: training')
    parser.add_argument('--train_file',type=str,default="/home/kihara/zhang038/Desktop/CS536_project/group_project/DeepQueueNet/data/4-port switch/FIFO/_hdf/train.h5")
    parser.add_argument('--test1_file',type=str,default="/home/kihara/zhang038/Desktop/CS536_project/group_project/DeepQueueNet/data/4-port switch/FIFO/_hdf/test1.h5")
    parser.add_argument('--test2_file',type=str,default="/home/kihara/zhang038/Desktop/CS536_project/group_project/DeepQueueNet/data/4-port switch/FIFO/_hdf/test2.h5")
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--save_dir',type=str, default='/net/kihara/scratch/zhang038/CS536_project/saved_models_l2/')
    parser.add_argument('--gpu',type=str,default='0',help='Choose gpu id, example: \'1,2\'(specify use gpu 1 and 2)')
    parser.add_argument('--weight_decay',type=float, default=0)
    args = parser.parse_args()
    params = vars(args)
    return params