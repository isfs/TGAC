import math
import logging
import time
import sys
import random
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np

from model.tgn import TGN
from utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP
from utils.data_processing import compute_time_statistics, get_data_node_classification
from evaluation.evaluation import eval_node_classification


from typing import Optional, Tuple
import augment as aug

from utils.data_processing import Data
import torch.nn.functional as F
from torch_geometric.utils import degree, to_undirected, to_networkx

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=100, help='Batch_size')
parser.add_argument('--prefix', type=str, default='tgn-attn', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=10, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--use_validation', action='store_true',
                    help='Whether to use a validation set')
parser.add_argument('--new_node', action='store_true', help='model new node')

parser.add_argument('--loss', type=str, default='cross', help='loss function')
parser.add_argument('--weight', type=int, default=10, help='loss weight')
parser.add_argument('--drop_scheme', type=str, default='pr', help='drop function')
parser.add_argument('--drop1', type=float, default=0.2, help='drop function')
parser.add_argument('--drop2', type=float, default=0.4, help='drop function')
parser.add_argument('--prune', type=float, default=0.05, help='prune function')
parser.add_argument('--decay', type=float, default=1e-5, help='decay function')


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

LOSS_FUNCTION = args.loss
DROP = args.drop_scheme

USE_MEMORY = True

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/new-{args.prefix}-{args.data}-{args.drop_scheme}-{args.loss}' + '\
  node-classification.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/new-{args.prefix}-{args.data}-{args.drop_scheme}-{args.loss}-{epoch}' + '\
  node-classification.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, last_update, edge_index, t, msg):
        aug1, aug2 = self.augmentor
        edge_index1, t1, msg1 = aug.drop_edge_weighted(edge_index, t, msg, drop_weights, p=0.3)
        edge_index2, t2, msg2 = aug.drop_edge_weighted(edge_index, t, msg, drop_weights, p=0.4)
        z = self.encoder(x, last_update, edge_index, t, msg)
        z1 = self.encoder(x, edge_index1, t1, msg1)
        z2 = self.encoder(x, edge_index2, t2, msg2)
        return z, z1, z2
    
class GRACE(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.num_hidden = num_hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        # ret = ret.mean() if mean else ret.sum()

        return ret
  


def save_node_embeddings(embeddings, path):
        writer = open(path, 'w')
        writer.write('%d %d\n' % (len(embeddings), args.node_dim))
        node_dim = 172
        for n_idx in range(1, len(embeddings)):
            writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')
        writer.close()


full_data, node_features, edge_features, train_data, val_data, test_data = \
  get_data_node_classification(DATA, use_validation=args.use_validation)

max_idx = max(full_data.unique_nodes)

# a = [0]*127
# node_embedding = a*(max_idx+1)
node_embedding = [[0 for col in range(172)] for row in range(max_idx+1)]


train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  results_path = "results/new-{}-{}-{}-{}-node-classification-{}-{}.pkl".format(args.prefix, args.data, args.drop_scheme, args.loss, args.prune,
                                                                i) # if i > 0 else "results/gca-{}-{}-node_classification.pkl".format(
    # args.prefix, args.loss)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator, n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message)

  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)
  
  logger.debug('Num of training instances: {}'.format(num_instance))
  logger.debug('Num of batches per epoch: {}'.format(num_batch))

  logger.info('Loading saved TGN model')
  model_path = f'./saved_models/new-{args.prefix}-{args.data}-{args.drop_scheme}-{args.loss}-{args.prune}.pth'
  tgn.load_state_dict(torch.load(model_path))
  tgn.eval()
  logger.info('TGN models loaded')
  logger.info('Start training node classification task')

  decoder = MLP(node_features.shape[1], drop=DROP_OUT)
  decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
  decoder = decoder.to(device)
  decoder_loss_criterion = torch.nn.BCELoss()

  val_aucs = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  for epoch in range(args.n_epoch):
    start_epoch = time.time()
    
    # Initialize memory of the model at each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    tgn = tgn.eval()
    decoder = decoder.train()
    loss = 0
    
    
    
    for k in range(num_batch):
      s_idx = k * BATCH_SIZE
      e_idx = min(num_instance, s_idx + BATCH_SIZE)

      sources_batch = train_data.sources[s_idx: e_idx]
      destinations_batch = train_data.destinations[s_idx: e_idx]
      timestamps_batch = train_data.timestamps[s_idx: e_idx]
      edge_idxs_batch = full_data.edge_idxs[s_idx: e_idx]
      labels_batch = train_data.labels[s_idx: e_idx]

      size = len(sources_batch)

      decoder_optimizer.zero_grad()
      with torch.no_grad():
        source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                     destinations_batch,
                                                                                     destinations_batch,
                                                                                     timestamps_batch,
                                                                                     edge_idxs_batch,
                                                                                     NUM_NEIGHBORS)
        for i in range(len(sources_batch)):
          node_embedding[sources_batch[i]] = source_embedding[i].cpu().numpy()
          
      labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
      pred = decoder(source_embedding).sigmoid()
      decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)
      decoder_loss.backward()
      decoder_optimizer.step()
      loss += decoder_loss.item()
    train_losses.append(loss / num_batch)

    val_auc, node_embedding = eval_node_classification(tgn, decoder, val_data, full_data.edge_idxs, BATCH_SIZE,
                                       n_neighbors=NUM_NEIGHBORS, node_embedding=node_embedding)
    val_aucs.append(val_auc)
    
    test_auc = val_aucs[-1]
    
    
    # break

    pickle.dump({
      "val_aps": val_aucs,
      "train_losses": train_losses,
      "epoch_times": [0.0],
      "new_nodes_val_aps": [],
    }, open(results_path, "wb"))

    logger.info(f'Epoch {epoch}: train loss: {loss / num_batch}, val auc: {val_auc}, time: {time.time() - start_epoch}')
  
  if args.use_validation:
    if early_stopper.early_stop_check(val_auc):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      break
    else:
      torch.save(decoder.state_dict(), get_checkpoint_path(epoch))

  if args.use_validation:
    logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
    best_model_path = get_checkpoint_path(early_stopper.best_epoch)
    decoder.load_state_dict(torch.load(best_model_path))
    logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
    decoder.eval()

    test_auc = eval_node_classification(tgn, decoder, test_data, full_data.edge_idxs, BATCH_SIZE,
                                        n_neighbors=NUM_NEIGHBORS)
  else:
    # If we are not using a validation set, the test performance is just the performance computed
    # in the last epoch
    # test_auc = val_aucs[-1]
    test_auc = np.amax(val_aucs)
    
  pickle.dump({
    "val_aps": val_aucs,
    "test_ap": test_auc,
    "train_losses": train_losses,
    "epoch_times": [0.0],
    "new_nodes_val_aps": [],
    "new_node_test_ap": 0,
  }, open(results_path, "wb"))

  logger.info(f'test auc: {test_auc}')
  
  path='./node_emb/{}.emb'.format(args.data)
  save_node_embeddings(node_embedding, path)
  
# 