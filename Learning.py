from torch_geometric.data import Data, HeteroData
import pandas as pd
from GNN import HeteroGNN
from Graph import batch_gen
import torch
import typing
from tqdm import tqdm



def train(model: HeteroGNN, data: HeteroData, optimizer: torch.optim, loss_fn: torch.nn.modules.loss) -> typing.Tuple[float, int, int]:
  batch_loss = 0

  model.train()
  out = model(data)

  optimizer.zero_grad()
  loss = loss_fn(out, data.y)
  batch_loss = loss.item()
  loss.backward()
  optimizer.step()

  prediction = out.argmax(dim=-1)
  correct = (prediction == data.y).sum().item()
  all = data.y.shape[0]

  return batch_loss, correct, all


@torch.no_grad()
def evaluate(model: HeteroGNN, data: HeteroData) -> typing.Tuple[int, int]:
  model.eval()

  # for child in model.children():
  #   for ii in range(len(child)):
  #       if type(child[ii]) == BatchNorm1d:
  #           child[ii].track_running_stats = False

  out = model(data)
  prediction = out.argmax(dim=-1)
  correct = (prediction == data.y).sum().item()
  all = data.y.shape[0]
  model.train()

  return correct, all


def continuousGrowingBatchMaker(df: pd.DataFrame, entities: dict, eval_window_size: int):
    weeks = []
    train_graphs = []
    eval_graphs = []
    for season, seasondf in df.groupby('season'):
        for w, weekdf in seasondf.groupby('week'):
            weeks.append(weekdf)


    first_batch = batch_gen(
        df.loc[
            weeks[0].index[0]: weeks[0].index[-1],
            :
        ],
        entities=entities, # must never be anything else
        messaging = [], 
        supervision= list(range(weeks[0].index[0], weeks[0].index[-1] + 1)), 
        remove_supervision_links=True
    )
    train_graphs.append(first_batch)

    for weeknumber, w in enumerate(tqdm(weeks)):
        train_end_point = weeknumber
        if train_end_point > 0:
            hd = batch_gen(
                df.loc[
                    weeks[0].index[0]: weeks[train_end_point].index[-1]
                ],
                entities=entities, # must never be anything else
                messaging = list(range(weeks[0].index[0], weeks[train_end_point - 1].index[-1] + 1)), 
                supervision= list(range(weeks[train_end_point].index[0], weeks[train_end_point].index[-1] + 1)), 
                remove_supervision_links=True
            )
            train_graphs.append(hd)

        
        eval_start_point = train_end_point + 1
        eval_end_point = eval_start_point + eval_window_size - 1
        #print('eval', eval_start_point, eval_end_point)
        if eval_end_point > len(weeks) - 1:
            eval_end_point = len(weeks) - 1
            #print('eval cor', eval_start_point, eval_end_point)
            
        if weeknumber == len(weeks) - 1:
            continue

        #if eval_start_point == 1:
        ehd = batch_gen(
            df.loc[
                weeks[0].index[0]: weeks[eval_end_point].index[-1]
            ],
            entities=entities, # must never be anything else
            messaging = list(range(weeks[0].index[0], weeks[train_end_point].index[-1] + 1)), 
            supervision= list(range(weeks[eval_start_point].index[0], weeks[eval_end_point].index[-1] + 1)), 
            remove_supervision_links=True
        )
        eval_graphs.append(ehd)
    
    return train_graphs, eval_graphs


def continuousGrowingTrainer(df: pd.DataFrame, entities: dict, model: HeteroGNN, optimizer: torch.optim, criterion: torch.nn.modules.loss, epochs: int, eval_window_size: int):
    train_graphs, eval_graphs = continuousGrowingBatchMaker(df, entities, eval_window_size)
    for epoch in range(epochs):
        tcorrect = 0
        tloss = 0.0
        ttotal = 0
        ecorrect = 0
        etotal = 0
        for i in range(len(eval_graphs)):
            itloss, itcorrect, ittotal = train(model, train_graphs[i], optimizer, criterion)
            tloss += itloss
            tcorrect += itcorrect
            ttotal += ittotal

            iecorrect, ietotal = evaluate(model, eval_graphs[i])
            ecorrect += iecorrect
            etotal += ietotal
        itloss, itcorrect, ittotal = train(model, train_graphs[-1], optimizer, criterion)
        tloss += itloss
        tcorrect += itcorrect
        ttotal += ittotal

        print(f'epoch: {epoch} - train loss {tloss / len(train_graphs)} - train acc: {tcorrect / ttotal} - eval acc: {ecorrect/ etotal}')


def continuousWindowBatchMaker(df: pd.DataFrame, entities: dict, train_window_size: int, eval_window_size: int):
    weeks = []
    train_graphs = []
    eval_graphs = []
    for season, seasondf in df.groupby('season'):
        for w, weekdf in seasondf.groupby('week'):
            weeks.append(weekdf)


    first_batch = batch_gen(
        df.loc[
            weeks[0].index[0]: weeks[0].index[-1],
            :
        ],
        entities=entities, # must never be anything else
        messaging = [], 
        supervision= list(range(weeks[0].index[0], weeks[0].index[-1] + 1)), 
        remove_supervision_links=True
    )
    train_graphs.append(first_batch)

    for weeknumber, w in enumerate(tqdm(weeks)):
        train_start_point = max(0, weeknumber - train_window_size + 1)
        train_end_point = weeknumber
        if train_end_point > 0:
            hd = batch_gen(
                df.loc[
                    weeks[train_start_point].index[0]: weeks[train_end_point].index[-1]
                ],
                entities=entities, # must never be anything else
                messaging = list(range(weeks[train_start_point].index[0], weeks[train_end_point - 1].index[-1] + 1)), 
                supervision= list(range(weeks[train_end_point].index[0], weeks[train_end_point].index[-1] + 1)), 
                remove_supervision_links=True
            )
            train_graphs.append(hd)

        
        eval_start_point = train_end_point + 1
        eval_end_point = eval_start_point + eval_window_size - 1
        #print('eval', eval_start_point, eval_end_point)
        if eval_end_point > len(weeks) - 1:
            eval_end_point = len(weeks) - 1
            #print('eval cor', eval_start_point, eval_end_point)
            
        if weeknumber == len(weeks) - 1:
            continue

        #if eval_start_point == 1:
        ehd = batch_gen(
            df.loc[
                weeks[train_start_point].index[0]: weeks[eval_end_point].index[-1]
            ],
            entities=entities, # must never be anything else
            messaging = list(range(weeks[train_start_point].index[0], weeks[train_end_point].index[-1] + 1)), 
            supervision= list(range(weeks[eval_start_point].index[0], weeks[eval_end_point].index[-1] + 1)), 
            remove_supervision_links=True
        )
        eval_graphs.append(ehd)
    
    return train_graphs, eval_graphs


def continuousWindowTrainer(df: pd.DataFrame, entities: dict, model: HeteroGNN, optimizer: torch.optim, criterion: torch.nn.modules.loss, epochs: int, train_window_size: int, eval_window_size: int):
    train_graphs, eval_graphs = continuousWindowBatchMaker(df, entities, train_window_size, eval_window_size)
    for epoch in range(epochs):
        tcorrect = 0
        tloss = 0.0
        ttotal = 0
        ecorrect = 0
        etotal = 0
        for i in range(len(eval_graphs)):
            itloss, itcorrect, ittotal = train(model, train_graphs[i], optimizer, criterion)
            tloss += itloss
            tcorrect += itcorrect
            ttotal += ittotal

            iecorrect, ietotal = evaluate(model, eval_graphs[i])
            ecorrect += iecorrect
            etotal += ietotal
        itloss, itcorrect, ittotal = train(model, train_graphs[-1], optimizer, criterion)
        tloss += itloss
        tcorrect += itcorrect
        ttotal += ittotal

        print(f'epoch: {epoch} - train loss {tloss / len(train_graphs)} - train acc: {tcorrect / ttotal} - eval acc: {ecorrect/ etotal}')


def continuousWindowTestBatchMaker(full_df: pd.DataFrame, test_df: pd.DataFrame, entities: dict, train_window_size:int, test_window_size: int):
    weeks = []
    test_graphs = []
    train_graphs = []

    for season, seasondf in test_df.groupby('season'):
        for w, weekdf in seasondf.groupby('week'):
            weeks.append(weekdf)

    for weeknumber, w in enumerate(tqdm(weeks)):
        test_start_idx = weeks[weeknumber].index[0] - (weeks[weeknumber].shape[0] * test_window_size)
        test_end_idx = weeks[weeknumber].index[-1]
        hd = batch_gen(
            full_df.loc[
                test_start_idx: test_end_idx
            ],
            entities=entities, # must never be anything else
            messaging = list(range(test_start_idx, weeks[weeknumber].index[0])), 
            supervision= list(range(weeks[weeknumber].index[0], weeks[weeknumber].index[-1] + 1)), 
            remove_supervision_links=True
        )
        test_graphs.append(hd)

        
        train_start_idx = weeks[weeknumber].index[0] - (weeks[weeknumber].shape[0] * train_window_size)
        train_end_idx = weeks[weeknumber].index[-1]
        
        thd = batch_gen(
            full_df.loc[
                train_start_idx: train_end_idx
            ],
            entities=entities, # must never be anything else
            messaging = list(range(train_start_idx, weeks[weeknumber].index[0])), 
            supervision= list(range(weeks[weeknumber].index[0], weeks[weeknumber].index[-1] + 1)), 
            remove_supervision_links=True
        )
        train_graphs.append(thd)
    
    return test_graphs, train_graphs


def continuousWindowTester(full_df: pd.DataFrame, test_df: pd.DataFrame, entities: dict, model: HeteroGNN, optimizer: torch.optim, criterion: torch.nn.modules.loss, epochs: int, train_window_size: int, test_window_size: int):
    test_graphs, train_graphs = continuousWindowTestBatchMaker(full_df, test_df, entities, train_window_size, test_window_size)
    tcorrect = 0
    ttotal = 0
    for i in range(len(test_graphs)):
        icorrect, itotal = evaluate(model, test_graphs[i])
        tcorrect += icorrect
        ttotal += itotal
        train(model, train_graphs[i], optimizer, criterion)
    print(f"Test Accuracy: {tcorrect / ttotal}")




