from typing import Literal, List, Union
import torch
import torch_geometric as pyg
import torch_geometric.data as pyg_data
from GNN import HeteroGNN
import GraphManager
import Config
import pickle


win = 0
loss = 0
tie = 0

def result_train_step(
    model: HeteroGNN,
    g: pyg_data.HeteroData,
    criterion,
    optimizer: torch.optim.Optimizer
):
    model.train()
    optimizer.zero_grad()

    out = model(g)
    loss = criterion(out, g.y.long())
    loss.backward()
    optimizer.step()

    pred = torch.argmax(out, dim=-1)
    correct = (pred == g.y).sum().item()
    total = g.y.shape[0]

    return loss.item(), correct, total


def goal_diff_train_step(
    model: HeteroGNN,
    g: pyg_data.HeteroData,
    criterion,
    optimizer: torch.optim.Optimizer
):
    model.train()
    optimizer.zero_grad()

    out = model(g).flatten()
    goal_diffs = g.hgoal - g.agoal
    loss = criterion(out, goal_diffs.float().to(Config.GLOBALS.DEVICE.value))
    loss.backward()
    optimizer.step()

    pred = torch.round(out).detach()
    pred = torch.round(out).detach()
    result_pred = pred.clone()
    result_pred[pred > 0] = 0
    result_pred[pred == 0] = 1
    result_pred[pred < 0] = 2
    correct = (result_pred == g.y).sum().item()
    total = g.y.shape[0]

    return loss.item(), correct, total


@torch.no_grad()
def result_evaluation(model: HeteroGNN, g: pyg_data.HeteroData):
    global win
    global loss
    global tie
    model.eval()

    out = model(g)
    pred = torch.argmax(out, dim=-1)
    correct = (pred == g.y).sum().item()
    total = g.y.shape[0]

    win += (pred == 0).sum().item()
    tie += (pred == 1).sum().item()
    loss += (pred == 2).sum().item()

    model.train()
    return correct, total


@torch.no_grad()
def goal_diff_evaluation(model: HeteroGNN, g: pyg_data.HeteroData):
    global win
    global loss
    global tie
    model.eval()

    out = model(g).flatten()
    pred = torch.round(out).detach()
    result_pred = pred.clone()
    result_pred[pred > 0] = 0
    result_pred[pred == 0] = 1
    result_pred[pred < 0] = 2
    correct = (result_pred == g.y).sum().item()
    total = g.y.shape[0]

    return correct, total



def evaluation(
    model: HeteroGNN,
    graph_list: List[pyg_data.HeteroData],
    eval_indcs: List[int],
    mode: Literal['RP', 'GDP'], #RP=Result Prediction - GDP=Goal Difference Prediction
    eval_acc_list: list=None,
    label: Literal['Validation', 'Test', None]=None,
    return_counts: bool=False
):
    if eval_indcs is None: return

    if mode == 'RP':
        eval_fn = result_evaluation
    elif mode == 'GDP':
        eval_fn = goal_diff_evaluation
    else:
        print(f'Error: mode must be in {["RP", "GDP"]}')
        return None

    t_correct = 0
    t_total = 0
    for idx in eval_indcs:
        g = graph_list[idx]
        correct, total = eval_fn(model, g)
        t_correct += correct
        t_total += total
    
    if label is not None:
        print(f'{label} Accuracy: {t_correct / t_total: .3f}')
    if label == 'Validation':
        eval_acc_list.append(t_correct / t_total)
    if return_counts:
        return t_correct, t_total
    return t_correct / t_total



def train(
    model: HeteroGNN,
    graph_list: List[pyg_data.HeteroData],
    train_indcs: List[int],
    eval_indcs: Union[List[int], None],
    mode: Literal['RP', 'GDP'], #RP=Result Prediction - GDP=Goal Difference Prediction
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    loss_list: list,
    train_acc_list: list,
    eval_acc_list: list
):
    if mode == 'RP':
        train_fn = result_train_step
    elif mode == 'GDP':
        train_fn = goal_diff_train_step
    else:
        print(f'Error: mode must be in {["RP", "GDP"]}')
        return None
    for epoch in range(epochs):
        t_loss = 0
        t_correct = 0
        t_total = 0
        for idx in train_indcs:
            g = graph_list[idx]
            loss, correct, total = train_fn(model, g, criterion, optimizer)
            t_loss += loss
            t_correct += correct
            t_total += total
        print(f'=================================== EPOCH {epoch + 1} ===================================')
        print(f'Average Loss: {t_loss / len(train_indcs)} - Train Accuracy: {t_correct / t_total: .3f}')
        loss_list.append(t_loss / len(train_indcs))
        train_acc_list.append(t_correct / t_total)

        evaluation(
            model,
            graph_list,
            eval_indcs,
            mode,
            eval_acc_list,
            'Validation'
        )

        if (epoch+1) % Config.GLOBALS.SaveEvery.value == 0:
            torch.save(model, f'{Config.GLOBALS.SavePath.value}model.pth')
            with open(f'{Config.GLOBALS.SavePath.value}lists.pl', 'wb') as pf:
                pickle.dump((loss_list, train_acc_list, eval_acc_list), pf)




def Phase1(dl, model, criterion, optimizer, train_fn, eval_fn, already_saved, rounds, epochs, loss_list, train_acc_list, eval_acc_list):
    for round in range(rounds):
        print(f'############################## Round {round + 1} ##############################')
        for league, league_df in dl.dataset.groupby('league'):
            print(f'Training On: {league}')
            if already_saved:
                gm = GraphManager.load(f'{Config.GLOBALS.LoadPath.value}{league}.gm')
            else:
                gm = GraphManager.load(f'{Config.GLOBALS.SavePath.value}{league}.gm')
            try:
                

                    pass

                    
            except KeyboardInterrupt:
                pass

            t_correct = 0
            t_total = 0
            for idx in gm.test_mask:
                g = gm.graph_list[idx]
                correct, total = eval_fn(model, g)
                t_correct += correct
                t_total += total
            print(f'Test Accuracy: {t_correct / t_total: .3f}')
            torch.save(model, f'{Config.GLOBALS.SavePath.value}model_{league}_R{round+1}_Ph1.pth')
        torch.save(model, f'{Config.GLOBALS.SavePath.value}model_All_R{round+1}_Ph1.pth')
            


def Phase2(dl, model, criterion, optimizer, train_fn, eval_fn, already_saved, rounds, epochs, loss_list, train_acc_list, eval_acc_list):
    for round in range(rounds):
        print(f'############################## Round {round + 1} ##############################')
        for league, league_df in dl.dataset.groupby('league'):
            print(f'Training On: {league}')
            if already_saved:
                gm = GraphManager.load(f'{Config.GLOBALS.LoadPath.value}{league}.gm')
            else:
                gm = GraphManager.load(f'{Config.GLOBALS.SavePath.value}{league}.gm')
            try:
                train_indcs = gm.train_mask + gm.validation_mask
                for epoch in range(epochs):
                    t_loss = 0
                    t_correct = 0
                    t_total = 0
                    
                    for idx in train_indcs:
                        g = gm.graph_list[idx]
                        loss, correct, total = train_fn(model, g, criterion, optimizer)
                        t_loss += loss
                        t_correct += correct
                        t_total += total
                    print(f'=================================== EPOCH {epoch + 1} ===================================')
                    print(f'Average Loss: {t_loss / len(gm.train_mask)} - Train Accuracy: {t_correct / t_total: .3f}')
                    loss_list.append(t_loss / len(gm.train_mask))
                    train_acc_list.append(t_correct / t_total)

                    t_correct = 0
                    t_total = 0


                    if (epoch+1) % Config.GLOBALS.SaveEvery.value == 0:
                        torch.save(model, f'{Config.GLOBALS.SavePath.value}model.pth')
                        with open(f'{Config.GLOBALS.SavePath.value}lists.pl', 'wb') as pf:
                            pickle.dump((loss_list, train_acc_list, eval_acc_list), pf)
                            
            except KeyboardInterrupt:
                pass
            t_correct = 0
            t_total = 0

            for idx in gm.test_mask:
                g = gm.graph_list[idx]
                correct, total = eval_fn(model, g)
                t_correct += correct
                t_total += total
            print(f'Test Accuracy: {t_correct / t_total: .3f}')
            torch.save(model, f'{Config.GLOBALS.SavePath.value}model_{league}_R{round+1}_Ph2.pth')
        torch.save(model, f'{Config.GLOBALS.SavePath.value}model_All_R{round+1}_Ph2.pth')



def Phase3(dl, model, criterion, optimizer, train_fn, eval_fn, already_saved, epochs, loss_list, train_acc_list, eval_acc_list):
    for league, league_df in dl.dataset.groupby('league'):
        print(f'Training On: {league}')
        if already_saved:
            gm = GraphManager.load(f'{Config.GLOBALS.LoadPath.value}{league}.gm')
        else:
            gm = GraphManager.load(f'{Config.GLOBALS.SavePath.value}{league}.gm')
        try:
            train_indcs = gm.train_mask + gm.validation_mask
            test_correct = 0
            test_total = 0
            for test_idx in gm.test_mask:
                print(f'--> Testing Week: {test_idx}')
                test_g = gm.graph_list[test_idx]
                correct, total = eval_fn(model, test_g)
                test_correct += correct
                test_total += total
                train_indcs.append(test_idx)

                for epoch in range(epochs):
                    t_loss = 0
                    t_correct = 0
                    t_total = 0
                    
                    for idx in train_indcs:
                        g = gm.graph_list[idx]
                        loss, correct, total = train_fn(model, g, criterion, optimizer)
                        t_loss += loss
                        t_correct += correct
                        t_total += total
                    print(f'=================================== EPOCH {epoch + 1} ===================================')
                    print(f'Average Loss: {t_loss / len(gm.train_mask)} - Train Accuracy: {t_correct / t_total: .3f}')
                    loss_list.append(t_loss / len(gm.train_mask))
                    train_acc_list.append(t_correct / t_total)

                    t_correct = 0
                    t_total = 0


                    if (epoch+1) % Config.GLOBALS.SaveEvery.value == 0:
                        torch.save(model, f'{Config.GLOBALS.SavePath.value}model.pth')
                        with open(f'{Config.GLOBALS.SavePath.value}lists.pl', 'wb') as pf:
                            pickle.dump((loss_list, train_acc_list, eval_acc_list), pf)
                            
        except KeyboardInterrupt:
            pass

        torch.save(model, f'{Config.GLOBALS.SavePath.value}model.pth')
        with open(f'{Config.GLOBALS.SavePath.value}lists.pl', 'wb') as pf:
            pickle.dump((loss_list, train_acc_list, eval_acc_list), pf)

        print(f'Test Accuracy: {test_correct / test_total: .3f}')
        torch.save(model, f'{Config.GLOBALS.SavePath.value}model_{league}_Ph3.pth')
    torch.save(model, f'{Config.GLOBALS.SavePath.value}model_All_Ph3.pth')






