import os
import torch
import cv2
import time
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PUIENet_MP import mynet
from data import get_eval_set


# settings
parser = argparse.ArgumentParser(description='PyTorch PUIE-Net')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use Default=123')
parser.add_argument('--device', type=str, default='cuda:2')
parser.add_argument('--input_dir', type=str, default='dataset/new_UIEBD/test/image')
parser.add_argument('--output', default='results/UIEBD', help='Location to save checkpoint models')
parser.add_argument('--reference_out', type=str, default='mp')
parser.add_argument('--model', default='weights/puie.pth', help='Pretrained base model')


opt = parser.parse_args()
print(opt)
device = torch.device(opt.device)
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# torch.manual_seed(opt.seed)
# if cuda:
#     torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = get_eval_set(opt.input_dir, opt.input_dir)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')

model = mynet(opt)

model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained model is loaded.')

if cuda:
    model = model.cuda(device)


def eval():
    model.eval()
    torch.set_grad_enabled(False)
    for batch in testing_data_loader:
        with torch.no_grad():
            input, _, name = Variable(batch[0]), Variable(batch[1]), batch[2]
        if cuda:
            input = input.cuda(device)

        with torch.no_grad():
            
            model.forward(input, input, training=False)
            t0 = time.time()
            prediction = model.sample(testing=True)
            t1 = time.time()
            print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            save_img_2(prediction.cpu().data, name[0], opt.reference_out)


def save_img_2(img, img_name, out):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    # save img
    save_dir = os.path.join(opt.output, out)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    name_list = img_name.split('.', 1)
    save_fn = save_dir + '/' + name_list[0] + '.' + name_list[1]
    cv2.imwrite(save_fn, cv2.cvtColor(save_img * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])

eval()
