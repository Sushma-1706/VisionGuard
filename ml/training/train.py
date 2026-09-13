"""Train a ResNet-18 classifier from an ImageFolder NEU-style dataset.
Expected splits: dataset/{train,val,test}/{normal,crack,inclusion,patches,pitted_surface,rolled-in_scale,scratches}.
"""
import argparse, json
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from backend.app.ml.model import CLASS_NAMES, create_model
def transforms(train: bool):
    steps=[v2.Resize((224,224))]
    if train: steps.append(v2.RandomHorizontalFlip())
    return v2.Compose(steps+[v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize((.485,.456,.406),(.229,.224,.225))])
def evaluate(model, loader, device):
    model.eval(); actual=[]; predicted=[]
    with torch.no_grad():
        for x,y in loader: actual+=y.tolist(); predicted+=model(x.to(device)).argmax(1).cpu().tolist()
    precision,recall,f1,_=precision_recall_fscore_support(actual,predicted,average="weighted",zero_division=0)
    return {"accuracy":accuracy_score(actual,predicted),"precision":precision,"recall":recall,"f1":f1}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True);p.add_argument("--epochs",type=int,default=10);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--learning-rate",type=float,default=1e-3);p.add_argument("--model",default="resnet18");p.add_argument("--output",type=Path,default=Path("ml/saved_models/visionguard_resnet18.pt"));args=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); train=ImageFolder(args.dataset/"train",transforms(True)); val=ImageFolder(args.dataset/"val",transforms(False))
    if train.classes != CLASS_NAMES: raise ValueError(f"Class folder order must be {CLASS_NAMES}; got {train.classes}")
    loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=2); vloader=DataLoader(val,batch_size=args.batch_size,num_workers=2)
    model=create_model(True).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate); criterion=nn.CrossEntropyLoss(); best=-1.; metrics={}
    for epoch in range(args.epochs):
        model.train()
        for x,y in loader: optimizer.zero_grad(); loss=criterion(model(x.to(device)),y.to(device)); loss.backward(); optimizer.step()
        metrics=evaluate(model,vloader,device); print(f"epoch {epoch+1}: {metrics}")
        if metrics["f1"]>best: args.output.parent.mkdir(parents=True,exist_ok=True); torch.save({"model_state_dict":model.state_dict(),"metrics":metrics,"class_names":CLASS_NAMES,"training_config":vars(args)},args.output);best=metrics["f1"]
    (args.output.with_suffix(".metrics.json")).write_text(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
