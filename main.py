import argparse
import torch 
from torch.utils.data import DataLoader

from utils.gcs_utils import get_gcs_info
from utils.utils import make_dirs
from data.dataset import Pix2PixMedicalImageDataset
from data.data_prepareing import get_data
from data.transform import transform
from trainer import Trainer
from tester import Tester

def main(args):
    print(args)
    torch.manual_seed(args.seed)
    print("[INFO] Welcome to my Zone")
    print("[INFO] Create directories")
    save_path = make_dirs(args.mode)
    df = get_data()
    
    mode = args.mode
    p_no = args.p_no
    
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio

    batch_size = args.batch_size
    num_workers = args.num_workers

    dataset = Pix2PixMedicalImageDataset(df=df, transform=transform, mode=mode, max_samples=100, fs=fs, p_no=p_no)

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    print(  "[INFO] data set has been splited."
            "\n     train size  :", len(train_dataset), 
            "\n     val size    :", len(val_dataset), 
            "\n     test size   :", len(test_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print('[INFO] Data has been loaded.')
    trainer = Trainer(args=args,save_path=save_path, train_loader=train_loader, val_loader=val_loader)
    print('[INFO] Train strats.')
    trainer.train()
    
    print('[INFO] Test strats.')
    tester = Tester(args=args, save_path=save_path, test_loader=test_loader)  # Tester 인스턴스 생성
    tester.test()  # 테스트 실행
    print('[INFO] THANK YOU.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # implement
    parser.add_argument('--mode', type=str, required=True,
                        choices=['L2P','P2L'], help='set mode')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for random number generator')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training data (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation data (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test data (default: 0.15)')

    # data
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers used in DataLoader')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--p_no', type=int, nargs=2, default=(10, 19), 
                        help='select range of folders (e.g., p10 to p19)')

    parser.add_argument('--L1_lambda', type=int, default=100, help='set L1_lambda')
    parser.add_argument('--lr', type=float, default=0.0002, help='set learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='set beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='set beta2')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='set device')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stop')	  
    
    # save path
    args = parser.parse_args()
    main(args)