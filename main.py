import argparse
from train import train_model
from predict import predict
from realtime import run_realtime_detection

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 安全帽与人体检测项目主入口")
    subparsers = parser.add_subparsers(dest='command', required=True, help="可用的命令")

    # --- 'train' 命令 ---
    parser_train = subparsers.add_parser('train', help='根据YAML配置文件训练一个新的模型')
    parser_train.add_argument('--config', type=str, required=True, help="指向数据集和训练参数的.yaml配置文件路径")

    # --- 'predict' 命令 ---
    parser_predict = subparsers.add_parser('predict', help='使用已训练好的模型进行图片或视频预测')
    parser_predict.add_argument('--model', type=str, required=True, help="指向.pt模型文件的路径")
    parser_predict.add_argument('--source', type=str, required=True, help="指向待预测的图片或视频文件")
    parser_predict.add_argument('--conf', type=float, default=0.5, help="检测结果的置信度阈值")

    # --- 'realtime' 命令 ---
    parser_realtime = subparsers.add_parser('realtime', help='启动摄像头进行实时检测')
    parser_realtime.add_argument('--model', type=str, required=True, help="指向.pt模型文件的路径")
    parser_realtime.add_argument('--camera-id', type=int, default=0, help="要使用的摄像头ID (通常为0)")
    parser_realtime.add_argument('--conf', type=float, default=0.5, help="检测结果的置信度阈值")

    args = parser.parse_args()

    if args.command == 'train':
        train_model(args.config)
    elif args.command == 'predict':
        predict(args.model, args.source, args.conf)
    elif args.command == 'realtime':
        run_realtime_detection(args.model, args.camera_id, args.conf)

if __name__ == '__main__':
    main()