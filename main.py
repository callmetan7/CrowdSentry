from src.preprocess import preprocess_data
from src.train import train
from src.evaluate import evaluate

preprocess_data('A')  # For Part A preprocess_data('B')  # For Part B
data_dir_A = "data/processed/part_A/train"
model_path = "mcnn_combined.pth"
data_dir_B = "data/processed/part_B/train"  

def main():
    train(
        data_dir_A=data_dir_A,
        data_dir_B=data_dir_B,
        model_path=model_path,
        epochs=20,
        batch_size=8,
        learning_rate=5e-5,
        validation_split=0.2
    )
    evaluate(
        modelPath=model_path,
        data_dir_A=data_dir_A,
        data_dir_B=data_dir_B,  # Set to None if not using Part_B
        batch_size=4
    )

if __name__ == "__main__":
    main()