import torch

def val_DQN(val_dataloader,device,model,loss_fn,epoch, val_log):
    loss_value = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_dataloader):
            X, y = sample
            batch_size = X.size(0)
            X,y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred,y)
            loss_value += loss.item()
            with open(val_log,'a+') as f:
                f.write(str((epoch-1)*batch_idx + batch_idx)+"\t"+str(loss.item())+"\n")
            print("Epoch:"+str(epoch)+"\tbatch:"+str(batch_idx)+"\tval_loss:"+str(loss.item()))
            break
    return loss_value / len(val_dataloader)