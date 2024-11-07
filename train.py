#
def training_loop(n_epochs, lstm_model, optimiser, loss_fn, X_train, y_train, X_test, y_test):
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    for epoch in range(n_epochs + 1):
        lstm_model.train()
        outputs = lstm_model.forward(X_train) 
        
        optimiser.zero_grad() 
        
        if epoch == 0:
            print(f"outputs shape: {outputs.shape}")
            print(f"y_train shape: {y_train.shape}")
            
        loss = loss_fn(outputs, y_train)
        
        loss.backward()
        
        optimiser.step()
        
        lstm_model.eval()
        test_preds = lstm_model(X_test)
        test_loss = loss_fn(test_preds, y_test)
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, train loss: {loss.item():.5f}, test loss: {test_loss.item():.5f}")