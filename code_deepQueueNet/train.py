def train_DQN(train_loader,device,model,loss_fn,optimizer,epoch,train_log):
    loss_value = 0.0
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        X, y = sample
        batch_size = X.size(0)
        X,y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with open(train_log,'a+') as f:
            f.write(str((epoch-1)*batch_idx + batch_idx)+"\t"+str(loss.item())+"\n")
        loss_value += loss.item()
        print("Epoch:"+str(epoch)+"\tbatch:"+str(batch_idx)+"\ttrain_loss:"+str(loss.item()))
    return loss_value / len(train_loader)


        

    





