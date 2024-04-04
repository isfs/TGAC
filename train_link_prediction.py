import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics, get_data1

from typing import Optional, Tuple
import augment as aug

from utils.data_processing import Data
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit CollegeMsg mooc enron uci)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=512, help='Batch_size')
parser.add_argument('--prefix', type=str, default='tgn-attn', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
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
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
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
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')

parser.add_argument('--loss', type=str, default='cross', help='loss function')
parser.add_argument('--weight', type=float, default=0.1, help='loss weight')
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
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory

MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

LOSS_FUNCTION = args.loss

USE_MEMORY = True

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/new-{args.prefix}-{args.data}-{args.drop_scheme}-{args.loss}-{args.prune}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/new-{args.prefix}-{args.data}-{args.drop_scheme}-{args.loss}-{args.prune}-{epoch}.pth'
    
### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
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
  


### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  results_path = "results/new-{}-{}-{}-{}-{}-{}-{}-{}-prune{}.pkl".format(args.prefix, args.data,args.drop_scheme, args.loss, i, args.drop1, args.drop2, args.decay, args.prune)
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
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep)
  criterion = torch.nn.BCELoss()
  tau = 0.3
  model = GRACE(tgn, 172, 64, tau).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=args.decay)
  tgn = tgn.to(device)
  model = model.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  
  
  if args.drop_scheme == 'degree':
    drop_weights = aug.degree_drop_weights([train_data.sources, train_data.destinations], train_data.timestamps).to(device)
  elif args.drop_scheme == 'pr':
    drop_weights = aug.pr_drop_weights([train_data.sources, train_data.destinations], train_data.timestamps, aggr='sink', k=200).to(device)
  elif args.drop_scheme == 'sp':
    drop_weights = aug.sp_drop_weights([train_data.sources, train_data.destinations], train_data.timestamps).to(device)
  elif args.drop_scheme == 'evc':
    drop_weights = aug.evc_drop_weights(train_data).to(device)
  elif args.drop_scheme == 'random':
    drop_weights = aug.random_drop_weights([train_data.sources, train_data.destinations]).to(device)
  else:
    drop_weights = None


  prune_train_data = aug.prune(train_data, drop_weights, args.prune)
  
  if args.drop_scheme == 'degree':
    drop_weights = aug.degree_drop_weights([prune_train_data.sources, prune_train_data.destinations], prune_train_data.timestamps).to(device)
  elif args.drop_scheme == 'pr':
    drop_weights = aug.pr_drop_weights([prune_train_data.sources, prune_train_data.destinations], prune_train_data.timestamps, aggr='sink', k=200).to(device)
  elif args.drop_scheme == 'sp':
    drop_weights = aug.sp_drop_weights([prune_train_data.sources, prune_train_data.destinations], prune_train_data.timestamps).to(device)
  elif args.drop_scheme == 'evc':
    drop_weights = aug.evc_drop_weights(prune_train_data).to(device)
  elif args.drop_scheme == 'random':
    drop_weights = aug.random_drop_weights([prune_train_data.sources, prune_train_data.destinations]).to(device)
  else:
    drop_weights = None

  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    
    sources1, destinations1, timestamps1, labels1, edge_idxs1 = aug.drop_edge_weighted(prune_train_data.sources, prune_train_data.destinations, prune_train_data.timestamps, prune_train_data.labels, prune_train_data.edge_idxs, drop_weights, args.drop1)
    train_data1 = Data(sources1, destinations1, timestamps1, edge_idxs1, labels1)
    train_rand_sampler1 = RandEdgeSampler(train_data1.sources, train_data1.destinations)
    
    sources2, destinations2, timestamps2, labels2, edge_idxs2 = aug.drop_edge_weighted(prune_train_data.sources, prune_train_data.destinations, prune_train_data.timestamps, prune_train_data.labels, prune_train_data.edge_idxs, drop_weights, args.drop2)
    train_data2 = Data(sources2, destinations2, timestamps2, edge_idxs2, labels2)
    train_rand_sampler2 = RandEdgeSampler(train_data2.sources, train_data2.destinations)
    
    num_instance1 = len(train_data1.sources)
    num_instance2 = len(train_data2.sources)
    num_instance = max(len(train_data1.sources), len(train_data2.sources))
    num_batch = math.ceil(num_instance / BATCH_SIZE)
    
    for k in range(0, num_batch, args.backprop_every):
      loss = 0
      loss1 = 0
      loss2 = 0
      # loss3 = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx1 = batch_idx * BATCH_SIZE
        end_idx1 = min(num_instance1, start_idx1 + BATCH_SIZE)
        sources_batch1, destinations_batch1 = train_data1.sources[start_idx1:end_idx1], \
                                            train_data1.destinations[start_idx1:end_idx1]
        edge_idxs_batch1 = train_data1.edge_idxs[start_idx1: end_idx1]
        timestamps_batch1 = train_data1.timestamps[start_idx1:end_idx1]

        size1 = len(sources_batch1)
        _, negatives_batch1 = train_rand_sampler1.sample(size1)

        with torch.no_grad():
          pos_label1 = torch.ones(size1, dtype=torch.float, device=device)
          neg_label1 = torch.zeros(size1, dtype=torch.float, device=device)

        model = model.train()
        if len(sources_batch1)!=0:
          pos_prob1, neg_prob1, source_embedding1, _ = model.encoder.compute_edge_probabilities(sources_batch1, destinations_batch1, negatives_batch1,
                                                              timestamps_batch1, edge_idxs_batch1, NUM_NEIGHBORS, 1)

          f = lambda x: torch.exp(x/0.1)
            
          if size1==1:
            pos_prob1 = pos_prob1.view(1)
            neg_prob1 = neg_prob1.view(1)
          else:
            pos_prob1 = pos_prob1.squeeze()
            neg_prob1 = neg_prob1.squeeze()
            
          
          loss1 += criterion(pos_prob1, pos_label1) + criterion(neg_prob1, neg_label1)
        
        
        
        start_idx2 = batch_idx * BATCH_SIZE
        end_idx2 = min(num_instance2, start_idx2 + BATCH_SIZE)
        
        sources_batch2, destinations_batch2 = train_data2.sources[start_idx2:end_idx2], \
                                            train_data2.destinations[start_idx2:end_idx2]
        if len(sources_batch2):
          edge_idxs_batch2 = train_data2.edge_idxs[start_idx2: end_idx2]
          timestamps_batch2 = train_data2.timestamps[start_idx2:end_idx2]

          size2 = len(sources_batch2)
          _, negatives_batch2 = train_rand_sampler2.sample(size2)

          with torch.no_grad():
            pos_label2 = torch.ones(size2, dtype=torch.float, device=device)
            neg_label2 = torch.zeros(size2, dtype=torch.float, device=device)

          pos_prob2, neg_prob2, source_embedding2, _ = model.encoder.compute_edge_probabilities(sources_batch2, destinations_batch2, negatives_batch2,
                                                              timestamps_batch2, edge_idxs_batch2, NUM_NEIGHBORS, 2)
          
          if size2==1:
            pos_prob2 = pos_prob2.view(1)
            neg_prob2 = neg_prob2.view(1)
          else:
            pos_prob2 = pos_prob2.squeeze()
            neg_prob2 = neg_prob2.squeeze()
          
          loss2 += criterion(pos_prob2, pos_label2) + criterion(neg_prob2, neg_label2)

        common_elements = np.intersect1d(sources_batch1, sources_batch2)
        indices1 = []
        for element in common_elements:
          index = None
          for i in range(len(sources_batch1) - 1, -1, -1):
              if sources_batch1[i] == element:
                index = i
                break
          indices1.append(index)
          
        indices2 = []
        for element in common_elements:
          index = None
          for i in range(len(sources_batch2) - 1, -1, -1):
            if sources_batch2[i] == element:
                index = i
                break
          indices2.append(index)
        loss3 = 0.0
        if len(common_elements) > 2:
          loss3 += model.loss(source_embedding1[indices1], source_embedding2[indices2])
          loss3 = loss3.mean()
          
        loss = (loss1 + loss2) + args.weight * loss3
        
      loss /= args.backprop_every
      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to the start of time
      if USE_MEMORY:
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    # Validation uses the full graph
    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      train_memory_backup = tgn.memory.backup_memory()

    val_ap, val_auc, val_acc, val_f1 = eval_edge_prediction(model=tgn,
                                                            negative_edge_sampler=val_rand_sampler,
                                                            data=val_data,
                                                            n_neighbors=NUM_NEIGHBORS)
    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test edges are
      # strictly later in time than validation edges)
      tgn.memory.restore_memory(train_memory_backup)

    # Validate on unseen nodes
    nn_val_ap, nn_val_auc, nn_val_acc, nn_val_f1 = eval_edge_prediction(model=tgn,
                                                                        negative_edge_sampler=val_rand_sampler,
                                                                        data=new_node_val_data,
                                                                        n_neighbors=NUM_NEIGHBORS)

    if USE_MEMORY:
      # Restore memory we had at the end of validation
      tgn.memory.restore_memory(val_memory_backup)

    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    # Save temporary results to disk
    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info(
      'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info(
      'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

    logger.info(
      'val acc: {}, new node val acc: {}'.format(val_acc, nn_val_acc))
    logger.info(
      'val f1: {}, new node val f1: {}'.format(val_f1, nn_val_f1))


    # Early stopping
    if early_stopper.early_stop_check(val_ap):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      tgn.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      break
    else:
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen
  # nodes
  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()

  ### Test
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  test_ap, test_auc, test_acc, test_f1 = eval_edge_prediction(model=tgn,
                                                              negative_edge_sampler=test_rand_sampler,
                                                              data=test_data,
                                                              n_neighbors=NUM_NEIGHBORS)

  if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)

  # Test on unseen nodes
  nn_test_ap, nn_test_auc, nn_test_acc, nn_test_f1 = eval_edge_prediction(model=tgn,
                                                                          negative_edge_sampler=nn_test_rand_sampler,
                                                                          data=new_node_test_data,
                                                                          n_neighbors=NUM_NEIGHBORS)

  logger.info(
    'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
  logger.info(
    'Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))

  logger.info(
    'Test statistics: Old nodes -- acc: {}, f1: {}'.format(test_acc, test_f1))
  logger.info(
    'Test statistics: New nodes -- acc: {}, f1: {}'.format(nn_test_acc, nn_test_f1))

  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "new_node_test_ap": nn_test_ap,
    "test_auc": test_auc,
    "new_node_test_auc": nn_test_auc,
    "test_acc": test_acc,
    "new_node_test_acc": nn_test_acc,
    "test_f1": test_f1,
    "new_node_test_f1": nn_test_f1,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  logger.info('Saving TGN model')
  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(val_memory_backup)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  logger.info('TGN model saved')

# python gca_link.py --use_memory

