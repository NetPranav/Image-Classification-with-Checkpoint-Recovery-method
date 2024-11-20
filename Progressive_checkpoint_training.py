# declaring the saving and loading path of the model
from pathlib import Path
from timeit import default_timer as timer
from tqdm import tqdm
import warnings

MODEL_PATH= Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)

# create model save path
# saving and loading model
MODEL_NAME = "CNN_Model_state_dict.pth"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME


# actual training code using progressive check point loading

warnings.filterwarnings("ignore", category=FutureWarning)
# load the save dict if available
try:
    checkpoint = torch.load(MODEL_SAVE_PATH)
    CNN_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_CNN.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    print("Checkpoint loaded successfully. Starting from epoch:", start_epoch)
except FileNotFoundError:
    print(f"No checkpoint found at {MODEL_SAVE_PATH}. Starting training from scratch...")
    start_epoch = 0

# setting up the start timer to check time taken by model to complete
train_time_start_model =timer()
indx = 2
epochs = 4
# starting the first loop
for i in tqdm(range(indx)):
    # starting the second loop
    for epoch in tqdm(range(epochs)):
      print(f"Epoch: {epoch} ...")
      train_model(model = CNN_model,
                  data_loader = train_dataloader,
                  loss_fn = loss_fn,
                  optimizer = optimizer_CNN,
                  accuracy_fn = accuracy_fn,
                  device = device)
      testing_model(model = CNN_model,
                    data_loader = test_dataloader,
                    loss_fn = loss_fn,
                    accuracy_fn = accuracy_fn,
                    device = device)
        
# saving and loading the model state dict
    torch.save({
        'model_state_dict': CNN_model.state_dict(),
        'optimizer_state_dict': optimizer_CNN.state_dict(),
    }, MODEL_SAVE_PATH)
    # loading model
    checkpoint = torch.load(MODEL_SAVE_PATH)
    CNN_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_CNN.load_state_dict(checkpoint['optimizer_state_dict'])
train_time_end_model = timer()
Train_time(start = train_time_start_model,
         end = train_time_end_model,
         device = device)
