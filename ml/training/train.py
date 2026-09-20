"""Train a ResNet-18 classifier from an ImageFolder NEU-style dataset.
Expected splits: dataset/{train,val,test}/{normal,crack,inclusion,patches,pitted_surface,rolled-in_scale,scratches}.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from backend.app.ml.model import CLASS_NAMES, create_model
def transforms(train: bool):
    steps=[v2.Resize((224,224))]
    if train: steps += [v2.RandomHorizontalFlip(), v2.RandomRotation(12), v2.ColorJitter(brightness=.15, contrast=.15), v2.GaussianBlur(3, sigma=(.1, 1.2))]
    return v2.Compose(steps+[v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize((.485,.456,.406),(.229,.224,.225))])
def evaluate(model, loader, device):
    model.eval(); actual=[]; predicted=[]
    scores=[]
    with torch.no_grad():
        for x,y in loader:
            logits=model(x.to(device)); actual+=y.tolist(); predicted+=logits.argmax(1).cpu().tolist(); scores.extend(torch.softmax(logits, 1).cpu().tolist())
    precision,recall,f1,_=precision_recall_fscore_support(actual,predicted,average="weighted",zero_division=0)
    macro_precision,macro_recall,macro_f1,_=precision_recall_fscore_support(actual,predicted,average="macro",zero_division=0)
    try: roc_auc=roc_auc_score(actual, scores, multi_class="ovr", labels=list(range(len(CLASS_NAMES))))
    except ValueError: roc_auc=float("nan")
    return {"accuracy":accuracy_score(actual,predicted),"precision_weighted":precision,"recall_weighted":recall,"f1_weighted":f1,"precision_macro":macro_precision,"recall_macro":macro_recall,"f1_macro":macro_f1,"roc_auc_ovr":roc_auc}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True);p.add_argument("--epochs",type=int,default=10);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--learning-rate",type=float,default=1e-3);p.add_argument("--model",default="resnet18");p.add_argument("--output",type=Path,default=Path("ml/saved_models/visionguard_resnet18.pt"));args=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); train=ImageFolder(args.dataset/"train",transforms(True)); val=ImageFolder(args.dataset/"val",transforms(False))
    if set(train.classes) != set(CLASS_NAMES): raise ValueError(f"Class folders must be {CLASS_NAMES}; got {train.classes}")
    if train.classes != CLASS_NAMES: raise ValueError(f"ImageFolder ordering differs from the model class order: {train.classes}")
    counts=np.bincount(train.targets, minlength=len(CLASS_NAMES)); class_weights=len(train) / (len(CLASS_NAMES) * counts)
    sample_weights=[class_weights[target] for target in train.targets]
    sampler=WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=2); vloader=DataLoader(val,batch_size=args.batch_size,num_workers=2)
    model=create_model(True).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate); criterion=nn.CrossEntropyLoss(weight=torch.tensor(class_weights,dtype=torch.float32,device=device)); best=-1.; metrics={}
    for epoch in range(args.epochs):
        model.train()
        for x,y in loader: optimizer.zero_grad(); loss=criterion(model(x.to(device)),y.to(device)); loss.backward(); optimizer.step()
        metrics=evaluate(model,vloader,device); print(f"epoch {epoch+1}: {metrics}")
        if metrics["f1_macro"]>best:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            training_config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
            torch.save({"model_state_dict":model.state_dict(),"metrics":metrics,"class_names":CLASS_NAMES,"training_config":training_config},args.output)
            best=metrics["f1_macro"]
    (args.output.with_suffix(".metrics.json")).write_text(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
