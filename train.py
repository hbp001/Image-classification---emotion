from utils import accuracy, AverageMeter
import wandb
def train(model, dataloader, criterion, optimizer, epoch=9999):

    acc1_meter = AverageMeter(name='accuracy top 1')
    acc5_meter = AverageMeter(name='accuracy top 5')
    loss_meter = AverageMeter(name='loss')
    n_iters = len(dataloader)
    model.train()
    for iter_idx, (images, labels) in enumerate(dataloader):
        
        images = images.cuda() # cuda:GPU 컴퓨팅용 통합 개발 환경
       # print(f"image : {images}")
        labels = labels.cuda()
       # print(f"labels : {labels}")
        optimizer.zero_grad() # 역전파 이전에 gradient를 0으로 만들어 줌
        outputs = model(images)
#        print(f"outputs = model(images) : {outputs}")
        loss = criterion(outputs, labels) # 손실 함수-출력이 정답으로부터 얼마나 떨어져 있는지 추정값 계산
       # print(f"loss = criterion(outputs, labels) : {loss}")
        loss.backward() # loss를 역방향에서 실행
        optimizer.step() # argument로 전달 받은 parameter 업데이트

        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        loss_meter.update(loss.item(), images.shape[0])
        acc1_meter.update(acc1[0], images.shape[0])
        acc5_meter.update(acc5[0], images.shape[0])

        print(f"[Epoch {epoch}] iter {iter_idx} / {n_iters}: \tLoss {loss_meter.val:.4f}({loss_meter.avg:.4f}) \tAcc top-1 {acc1_meter.val:.2f}({acc1_meter.avg:.2f}) \tAcc top-5 {acc5_meter.val:.2f}({acc5_meter.avg:.2f})", end='\r')
    print("")
    print(f"Epoch {epoch} training finished")
    wandb.log({
        "loss": loss,
        "acc1": acc1,
        "acc5": acc5
    })
