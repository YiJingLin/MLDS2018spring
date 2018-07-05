def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    
    ### common arguments
    parser.add_argument('--weight_path', type=str, default='./weight/', 
                    help='pre-trained model\'s weight path')
    parser.add_argument('--history_path', type=str, default='./log/', 
                    help='save learning record path')
    parser.add_argument('--model_name', type=str, default='policy_weights.pkl', 
                    help='model weights name')
    parser.add_argument('--history_name', type=str, default='hisotry.pkl', 
                    help='learning history name')
    parser.add_argument('--n_episode', type=int, default=2000, 
                    help='number of episodes')
    parser.add_argument('--n_step', type=int, default=10000, 
                    help='number of step in each episode')
    parser.add_argument('--variance_reduction', action='store_true',
    				help='implement varaince reduction or not')

    ### PG arguments
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99')
    parser.add_argument('--decay_rate', type=float, default=0.99, metavar='G',
                    help='decay rate for RMSprop (default: 0.99)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='G',
                    help='Every how many episodes to da a param update')
    parser.add_argument('--seed', type=int, default=666, metavar='N',
                    help='random seed (default: 666)')
    
    ### DQN arguments


    return parser
